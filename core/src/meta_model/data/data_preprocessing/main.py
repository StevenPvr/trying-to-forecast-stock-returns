from __future__ import annotations

"""Preprocessing orchestrator: target construction, feature normalisation, split assignment."""

import logging
import importlib
import importlib.util
import sys
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_numba_spec = importlib.util.find_spec("numba")
NUMBA_AVAILABLE: bool = _numba_spec is not None
DecoratedFn = TypeVar("DecoratedFn", bound=Callable[..., Any])


class NumbaDecorator(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Callable[[DecoratedFn], DecoratedFn]:
        ...


if _numba_spec is not None:  # pragma: no branch
    _numba = importlib.import_module("numba")
    njit: NumbaDecorator = cast(NumbaDecorator, _numba.njit)
    prange: Callable[..., Any] = cast(Callable[..., Any], _numba.prange)
else:  # pragma: no cover - exercised when numba is unavailable locally
    def _njit_fallback(*args: Any, **kwargs: Any) -> Callable[[DecoratedFn], DecoratedFn]:
        def decorator(fn: DecoratedFn) -> DecoratedFn:
            return fn
        return decorator

    def _prange_fallback(*args: int) -> range:
        return range(*args)

    njit = _njit_fallback
    prange = _prange_fallback

from core.src.meta_model.data.constants import (
    DATASET_SPLIT_TEST_START_DATE,
    DATASET_SPLIT_VAL_FRACTION_OF_PRE_TEST_UNIQUE_DATES,
    RANDOM_SEED,
    SAMPLE_FRAC,
)
from core.src.meta_model.data.paths import (
    DATA_PREPROCESSING_DIR,
    FEATURES_OUTPUT_PARQUET,
    PREPROCESSED_OUTPUT_PARQUET,
    PREPROCESSED_OUTPUT_SAMPLE_CSV,
    PREPROCESSED_FEATURE_REGISTRY_JSON,
    PREPROCESSED_FEATURE_REGISTRY_PARQUET,
    PREPROCESSED_FEATURE_SCHEMA_MANIFEST_JSON,
    PREPROCESSED_RESEARCH_LABEL_PANEL_PARQUET,
    PREPROCESSED_TEST_PARQUET,
    PREPROCESSED_TEST_SAMPLE_CSV,
    PREPROCESSED_TRAIN_PARQUET,
    PREPROCESSED_TRAIN_SAMPLE_CSV,
    PREPROCESSED_VAL_PARQUET,
    PREPROCESSED_VAL_SAMPLE_CSV,
)
from core.src.meta_model.data.registry import (
    SAFE_FFILL_MAX_DAYS_COLUMN,
    FEATURE_NAME_COLUMN,
    build_feature_registry,
    build_feature_registry_from_columns,
    save_feature_registry,
    save_feature_schema_manifest,
)
from core.src.meta_model.model_contract import (
    EXECUTION_LAG_DAYS,
    HOLD_PERIOD_DAYS,
    LABEL_EMBARGO_DAYS,
    INTRADAY_BENCHMARK_RETURN_COLUMN,
    INTRADAY_CS_RANK_TARGET_COLUMN,
    INTRADAY_CS_ZSCORE_TARGET_COLUMN,
    INTRADAY_EXCESS_RETURN_COLUMN,
    INTRADAY_NET_RETURN_COLUMN,
    MEDIUM_HOLD_GROSS_RETURN_COLUMN,
    MEDIUM_HOLD_NET_RETURN_COLUMN,
    LEGACY_TARGET_COLUMN,
    MODEL_TARGET_COLUMN as CONTRACT_MODEL_TARGET_COLUMN,
    OVERNIGHT_NET_RETURN_COLUMN,
    REALIZED_RETURN_COLUMN,
    SHORT_HOLD_NET_RETURN_COLUMN,
    INTRADAY_SECTOR_RESIDUAL_RETURN_COLUMN,
    INTRADAY_GROSS_RETURN_COLUMN,
    WEEK_HOLD_BENCHMARK_RETURN_COLUMN,
    WEEK_HOLD_CS_RANK_TARGET_COLUMN,
    WEEK_HOLD_CS_ZSCORE_TARGET_COLUMN,
    WEEK_HOLD_EXCESS_RETURN_COLUMN,
    WEEK_HOLD_GROSS_RETURN_COLUMN,
    WEEK_HOLD_NET_RETURN_COLUMN,
    WEEK_HOLD_SECTOR_RESIDUAL_RETURN_COLUMN,
)

LOGGER: logging.Logger = logging.getLogger(__name__)

PREPROCESSING_START_DATE: date = date(2009, 1, 1)
COVID_START_DATE: date = date(2020, 2, 1)
COVID_END_DATE: date = date(2021, 12, 31)
DateLike = str | date | datetime
TARGET_COLUMN: str = LEGACY_TARGET_COLUMN
MODEL_TARGET_COLUMN: str = CONTRACT_MODEL_TARGET_COLUMN
SPLIT_COLUMN: str = "dataset_split"
TARGET_HORIZON_DAYS: int = HOLD_PERIOD_DAYS
TARGET_EXECUTION_LAG_DAYS: int = EXECUTION_LAG_DAYS
FEATURE_SAMPLE_FRAC: float = 0.5
FEATURE_SAMPLE_MAX_ROWS: int = 2000
PEARSON_PRESCREENER_THRESHOLD: float = 0.9
DISTANCE_CORRELATION_THRESHOLD: float = 0.95
TARGET_RELATED_COLUMNS: tuple[str, ...] = (
    TARGET_COLUMN,
    REALIZED_RETURN_COLUMN,
    INTRADAY_GROSS_RETURN_COLUMN,
    INTRADAY_NET_RETURN_COLUMN,
    INTRADAY_BENCHMARK_RETURN_COLUMN,
    INTRADAY_EXCESS_RETURN_COLUMN,
    INTRADAY_SECTOR_RESIDUAL_RETURN_COLUMN,
    INTRADAY_CS_ZSCORE_TARGET_COLUMN,
    INTRADAY_CS_RANK_TARGET_COLUMN,
    WEEK_HOLD_NET_RETURN_COLUMN,
    WEEK_HOLD_BENCHMARK_RETURN_COLUMN,
    WEEK_HOLD_EXCESS_RETURN_COLUMN,
    WEEK_HOLD_SECTOR_RESIDUAL_RETURN_COLUMN,
    WEEK_HOLD_CS_ZSCORE_TARGET_COLUMN,
    WEEK_HOLD_CS_RANK_TARGET_COLUMN,
    OVERNIGHT_NET_RETURN_COLUMN,
    SHORT_HOLD_NET_RETURN_COLUMN,
    MEDIUM_HOLD_GROSS_RETURN_COLUMN,
    MEDIUM_HOLD_NET_RETURN_COLUMN,
)
def load_feature_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input parquet not found: {path}")
    df: pd.DataFrame = pd.read_parquet(path)
    LOGGER.info("Loaded feature dataset: %d rows x %d cols", len(df), len(df.columns))
    return df


def filter_from_start_date(
    data: pd.DataFrame,
    start_date: DateLike = PREPROCESSING_START_DATE,
) -> pd.DataFrame:
    filtered: pd.DataFrame = data.copy()
    start_timestamp = pd.Timestamp(start_date)
    filtered["date"] = pd.to_datetime(filtered["date"])
    filtered = pd.DataFrame(filtered.loc[filtered["date"] >= start_timestamp].copy())
    LOGGER.info(
        "Filtered dataset from %s onward: %d rows x %d cols",
        start_timestamp.date(),
        len(filtered),
        len(filtered.columns),
    )
    return filtered.sort_values(["date", "ticker"]).reset_index(drop=True)


def create_target_main_group(
    group: pd.DataFrame,
    *,
    horizon_days: int = TARGET_HORIZON_DAYS,
    execution_lag_days: int = TARGET_EXECUTION_LAG_DAYS,
) -> pd.DataFrame:
    """Attach forward returns with decision at close(J).

    The primary week label follows the project contract: open(J+1) -> close(J+6)
    when ``execution_lag_days == 1`` and ``horizon_days == 5``.

    Net-of-cost columns match gross returns: cash-equity convention with no
    broker round-trip model (CFD fee plumbing removed).
    """
    ticker_group: pd.DataFrame = group.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    if ticker_group.empty:
        return ticker_group

    required_columns = {"stock_open_price", "stock_close_price"}
    missing_columns = required_columns.difference(ticker_group.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s) for target creation: {missing_list}")

    entry_open = pd.Series(ticker_group["stock_open_price"].shift(-execution_lag_days))
    entry_close = pd.Series(ticker_group["stock_close_price"].shift(-execution_lag_days))
    next_open = pd.Series(ticker_group["stock_open_price"].shift(-(execution_lag_days + 1)))
    short_exit_close = pd.Series(ticker_group["stock_close_price"].shift(-(execution_lag_days + 1)))
    medium_exit_open = pd.Series(
        ticker_group["stock_open_price"].shift(-(execution_lag_days + horizon_days)),
    )
    week_exit_close = pd.Series(
        ticker_group["stock_close_price"].shift(-(execution_lag_days + horizon_days)),
    )
    intraday_gross_return = np.log(entry_close / entry_open)
    overnight_gross_return = np.log(next_open / entry_close)
    short_hold_gross_return = np.log(short_exit_close / entry_open)
    medium_hold_gross_return = np.log(medium_exit_open / entry_open)
    week_hold_gross_return = np.log(week_exit_close / entry_open)

    zero_cost = pd.Series(0.0, index=ticker_group.index, dtype=np.float64)
    intraday_cost = zero_cost
    overnight_cost = zero_cost
    short_hold_cost = zero_cost
    medium_hold_cost = zero_cost
    week_hold_cost = zero_cost

    ticker_group[INTRADAY_GROSS_RETURN_COLUMN] = intraday_gross_return
    ticker_group[TARGET_COLUMN] = week_hold_gross_return
    ticker_group[REALIZED_RETURN_COLUMN] = week_hold_gross_return
    ticker_group[INTRADAY_NET_RETURN_COLUMN] = intraday_gross_return - intraday_cost
    ticker_group[OVERNIGHT_NET_RETURN_COLUMN] = overnight_gross_return - overnight_cost
    ticker_group[SHORT_HOLD_NET_RETURN_COLUMN] = short_hold_gross_return - short_hold_cost
    ticker_group[MEDIUM_HOLD_GROSS_RETURN_COLUMN] = medium_hold_gross_return
    ticker_group[MEDIUM_HOLD_NET_RETURN_COLUMN] = medium_hold_gross_return - medium_hold_cost
    ticker_group[WEEK_HOLD_GROSS_RETURN_COLUMN] = week_hold_gross_return
    ticker_group[WEEK_HOLD_NET_RETURN_COLUMN] = week_hold_gross_return - week_hold_cost
    return ticker_group


def _build_sector_key(data: pd.DataFrame) -> pd.Series:
    sector_series = data.get("company_sector", pd.Series(index=data.index, dtype="object"))
    industry_series = data.get("company_industry", pd.Series(index=data.index, dtype="object"))
    return pd.Series(
        sector_series.fillna(industry_series).fillna("__missing_sector__"),
        index=data.index,
    )


def _build_cross_sectional_metrics(
    data: pd.DataFrame,
    *,
    net_return_column: str,
    benchmark_column: str,
    excess_column: str,
    sector_residual_column: str,
    zscore_column: str,
    rank_column: str,
) -> pd.DataFrame:
    benchmark_forward_return = data.groupby("date")[net_return_column].transform("mean")
    excess_forward_return = data[net_return_column] - benchmark_forward_return
    daily_std = data.groupby("date")[net_return_column].transform("std")
    safe_daily_std = pd.Series(daily_std.replace(0.0, np.nan))
    cs_rank = data.groupby("date")[net_return_column].rank(method="average", pct=True) - 0.5
    sector_key = _build_sector_key(data)
    sector_mean = data.groupby([pd.to_datetime(data["date"]), sector_key])[net_return_column].transform(
        "mean",
    )

    return pd.DataFrame(
        {
            "date": pd.to_datetime(data["date"]),
            "ticker": data["ticker"].astype(str),
            benchmark_column: benchmark_forward_return,
            excess_column: excess_forward_return,
            sector_residual_column: data[net_return_column] - sector_mean,
            zscore_column: (
                (excess_forward_return / safe_daily_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            ),
            rank_column: pd.Series(cs_rank).fillna(0.0),
        },
    )


def build_target_metric_panel(data: pd.DataFrame) -> pd.DataFrame:
    intraday_metrics = _build_cross_sectional_metrics(
        data,
        net_return_column=INTRADAY_NET_RETURN_COLUMN,
        benchmark_column=INTRADAY_BENCHMARK_RETURN_COLUMN,
        excess_column=INTRADAY_EXCESS_RETURN_COLUMN,
        sector_residual_column=INTRADAY_SECTOR_RESIDUAL_RETURN_COLUMN,
        zscore_column=INTRADAY_CS_ZSCORE_TARGET_COLUMN,
        rank_column=INTRADAY_CS_RANK_TARGET_COLUMN,
    )
    week_metrics = _build_cross_sectional_metrics(
        data,
        net_return_column=WEEK_HOLD_NET_RETURN_COLUMN,
        benchmark_column=WEEK_HOLD_BENCHMARK_RETURN_COLUMN,
        excess_column=WEEK_HOLD_EXCESS_RETURN_COLUMN,
        sector_residual_column=WEEK_HOLD_SECTOR_RESIDUAL_RETURN_COLUMN,
        zscore_column=WEEK_HOLD_CS_ZSCORE_TARGET_COLUMN,
        rank_column=WEEK_HOLD_CS_RANK_TARGET_COLUMN,
    )
    merged = intraday_metrics.merge(
        week_metrics,
        on=["date", "ticker"],
        how="inner",
    )
    return pd.DataFrame(merged)


def apply_target_metric_panel(
    data: pd.DataFrame,
    metric_panel: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "date",
        "ticker",
        INTRADAY_BENCHMARK_RETURN_COLUMN,
        INTRADAY_EXCESS_RETURN_COLUMN,
        INTRADAY_SECTOR_RESIDUAL_RETURN_COLUMN,
        INTRADAY_CS_ZSCORE_TARGET_COLUMN,
        INTRADAY_CS_RANK_TARGET_COLUMN,
        WEEK_HOLD_BENCHMARK_RETURN_COLUMN,
        WEEK_HOLD_EXCESS_RETURN_COLUMN,
        WEEK_HOLD_SECTOR_RESIDUAL_RETURN_COLUMN,
        WEEK_HOLD_CS_ZSCORE_TARGET_COLUMN,
        WEEK_HOLD_CS_RANK_TARGET_COLUMN,
    ]
    merged = data.drop(
        columns=[column for column in metric_columns[2:] if column in data.columns],
        errors="ignore",
    ).merge(
        metric_panel.loc[:, metric_columns],
        on=["date", "ticker"],
        how="left",
        sort=False,
    )
    return pd.DataFrame(merged)


def create_target_main(
    data: pd.DataFrame,
    horizon_days: int = TARGET_HORIZON_DAYS,
    execution_lag_days: int = TARGET_EXECUTION_LAG_DAYS,
) -> pd.DataFrame:
    enriched: pd.DataFrame = data.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    target_parts: list[pd.DataFrame] = []
    for _, group in enriched.groupby("ticker", sort=False):
        target_parts.append(
            create_target_main_group(
                group,
                horizon_days=horizon_days,
                execution_lag_days=execution_lag_days,
            ),
        )

    result: pd.DataFrame = pd.concat(target_parts, ignore_index=True)
    result = apply_target_metric_panel(result, build_target_metric_panel(result))
    LOGGER.info(
        "Created forward-hold labels (zero frictions in net columns): "
        "execution_lag_days=%d | week_hold_sessions=%d "
        "(signal-day close decision, next-open entry -> close five sessions later).",
        execution_lag_days,
        horizon_days,
    )
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)


def exclude_covid_period(
    data: pd.DataFrame,
    covid_start_date: DateLike = COVID_START_DATE,
    covid_end_date: DateLike = COVID_END_DATE,
    target_horizon_days: int = TARGET_HORIZON_DAYS,
) -> pd.DataFrame:
    filtered = data.copy()
    dates = pd.to_datetime(filtered["date"])
    covid_start_timestamp = pd.Timestamp(covid_start_date)
    covid_end_timestamp = pd.Timestamp(covid_end_date)
    pre_covid_dates = pd.Index(
        pd.to_datetime(filtered.loc[dates < covid_start_timestamp, "date"]).drop_duplicates().sort_values(),
    )
    bridge_dates_before_covid: set[pd.Timestamp] = set()
    if target_horizon_days > 0:
        bridge_dates_before_covid = {
            pd.Timestamp(value)
            for value in pre_covid_dates[-target_horizon_days:].tolist()
        }
    exclusion_mask = dates.between(covid_start_timestamp, covid_end_timestamp) | dates.isin(bridge_dates_before_covid)
    result = pd.DataFrame(filtered.loc[~exclusion_mask].copy())
    LOGGER.info(
        "Excluded Covid period %s -> %s plus %d bridge dates before the cutoff (%d -> %d rows).",
        covid_start_timestamp.date(),
        covid_end_timestamp.date(),
        len(bridge_dates_before_covid),
        len(filtered),
        len(result),
    )
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)


def assign_dataset_splits(data: pd.DataFrame) -> pd.DataFrame:
    """Assign train/val/test using a fixed test start date and proportional train/val.

    Rows with ``date >= DATASET_SPLIT_TEST_START_DATE`` are test. The last
    ``LABEL_EMBARGO_DAYS`` business days strictly before that start are dropped
    (purge). Among remaining unique trading days before that purge window,
    approximately ``DATASET_SPLIT_VAL_FRACTION_OF_PRE_TEST_UNIQUE_DATES`` are
    validation (latest dates) and the rest are train, with an additional
    ``LABEL_EMBARGO_DAYS`` business-day purge between the last train day and
    the first validation day when there is enough history.

    Train thus receives ~70% of that pre-test timeline when val is 30% and test
    is date-based (the slack beyond 60/30 goes to train).
    """
    split_ready: pd.DataFrame = data.copy()
    dates: pd.Series = pd.to_datetime(split_ready["date"])
    test_start_ts: pd.Timestamp = pd.Timestamp(DATASET_SPLIT_TEST_START_DATE)
    embargo: int = LABEL_EMBARGO_DAYS
    test_embargo_cutoff: pd.Timestamp = test_start_ts - pd.offsets.BusinessDay(embargo)

    split_ready[SPLIT_COLUMN] = pd.Series(pd.NA, index=split_ready.index, dtype="object")

    split_ready.loc[dates >= test_start_ts, SPLIT_COLUMN] = "test"
    purge_before_test_mask: pd.Series = (dates >= test_embargo_cutoff) & (dates < test_start_ts)
    split_ready.loc[purge_before_test_mask, SPLIT_COLUMN] = pd.NA

    eligible_mask: pd.Series = dates < test_embargo_cutoff
    eligible_dates: pd.Index = pd.Index(
        pd.to_datetime(dates.loc[eligible_mask]).drop_duplicates().sort_values(),
    )
    n: int = int(len(eligible_dates))
    train_dates_list: list[pd.Timestamp]
    purge_mid_list: list[pd.Timestamp]
    val_dates_list: list[pd.Timestamp]

    if n == 0:
        train_dates_list = []
        purge_mid_list = []
        val_dates_list = []
    else:
        n_val: int = max(
            1,
            int(round(DATASET_SPLIT_VAL_FRACTION_OF_PRE_TEST_UNIQUE_DATES * n)),
        )
        split_at: int = n - n_val
        if split_at <= embargo:
            LOGGER.warning(
                "Only %d unique pre-test dates; cannot reserve %d-day train/val embargo — "
                "using contiguous train/val split.",
                n,
                embargo,
            )
            train_dates_list = eligible_dates[:split_at].tolist()
            purge_mid_list = []
            val_dates_list = eligible_dates[split_at:].tolist()
        else:
            train_last_idx: int = split_at - embargo - 1
            train_dates_list = eligible_dates[: train_last_idx + 1].tolist()
            purge_mid_list = eligible_dates[train_last_idx + 1 : split_at].tolist()
            val_dates_list = eligible_dates[split_at:].tolist()

    split_ready.loc[dates.isin(train_dates_list), SPLIT_COLUMN] = "train"
    split_ready.loc[dates.isin(val_dates_list), SPLIT_COLUMN] = "val"
    split_ready.loc[dates.isin(purge_mid_list), SPLIT_COLUMN] = pd.NA

    split_ready = pd.DataFrame(split_ready.loc[split_ready[SPLIT_COLUMN].notna()].copy())
    LOGGER.info(
        "Assigned dataset splits: train=%d, val=%d, test=%d (test_start=%s, pre_test_unique=%d)",
        int((split_ready[SPLIT_COLUMN] == "train").sum()),
        int((split_ready[SPLIT_COLUMN] == "val").sum()),
        int((split_ready[SPLIT_COLUMN] == "test").sum()),
        test_start_ts.date(),
        n,
    )
    return split_ready.sort_values(["date", "ticker"]).reset_index(drop=True)


def remove_rows_with_missing_values(
    data: pd.DataFrame,
    required_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    columns_to_check: list[str]
    if required_columns is None:
        columns_to_check = list(data.columns)
    else:
        columns_to_check = [column for column in required_columns if column in data.columns]
    if not columns_to_check:
        return data.sort_values(["date", "ticker"]).reset_index(drop=True)

    missing_rows_mask: pd.Series = pd.Series(data[columns_to_check].isna().any(axis=1))
    dropped_rows: int = int(missing_rows_mask.sum())
    cleaned: pd.DataFrame = data.loc[~missing_rows_mask].copy()
    if dropped_rows > 0:
        LOGGER.info(
            "Dropped %d rows with missing values in required columns before saving (%d -> %d).",
            dropped_rows,
            len(data),
            len(cleaned),
        )
    return cleaned.sort_values(["date", "ticker"]).reset_index(drop=True)


def drop_columns_with_missing_values(
    data: pd.DataFrame,
    protected_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    protected: set[str] = set(protected_columns or [])
    missing_by_column: pd.Series = data.isna().sum()
    columns_to_drop: list[str] = [
        str(column)
        for column, missing_count in missing_by_column.items()
        if column not in protected and int(missing_count) > 0
    ]
    if not columns_to_drop:
        return data.copy()

    cleaned: pd.DataFrame = data.drop(columns=columns_to_drop)
    LOGGER.info(
        "Dropped %d columns with missing values before saving.",
        len(columns_to_drop),
    )
    return cleaned


def drop_fully_missing_feature_columns(
    data: pd.DataFrame,
    protected_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    protected: set[str] = set(protected_columns or [])
    columns_to_drop: list[str] = [
        str(column)
        for column in data.columns
        if column not in protected and int(data[column].notna().sum()) == 0
    ]
    if not columns_to_drop:
        return data.copy()

    cleaned: pd.DataFrame = data.drop(columns=columns_to_drop)
    LOGGER.info(
        "Dropped %d feature columns that were fully missing after preprocessing.",
        len(columns_to_drop),
    )
    return cleaned


def build_feature_fill_limits(feature_registry: pd.DataFrame) -> dict[str, int | None]:
    registry = feature_registry.loc[:, [FEATURE_NAME_COLUMN, SAFE_FFILL_MAX_DAYS_COLUMN]].copy()
    return {
        str(feature_name): (
            None
            if pd.isna(limit_value)
            else int(limit_value)
        )
        for feature_name, limit_value in registry.itertuples(index=False)
    }


def forward_fill_features_by_ticker(
    data: pd.DataFrame,
    feature_fill_limits: dict[str, int | None] | None = None,
    protected_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    protected: set[str] = set(protected_columns or [])
    feature_columns: list[str] = [
        column for column in data.columns if column not in protected
    ]
    if not feature_columns:
        return data.sort_values(["date", "ticker"]).reset_index(drop=True)

    ordered = pd.DataFrame(data.sort_values(["ticker", "date"]).reset_index(drop=True))
    missing_before = int(ordered[feature_columns].isna().sum().sum())
    resolved_fill_limits = (
        feature_fill_limits
        if feature_fill_limits is not None
        else build_feature_fill_limits(build_feature_registry(ordered))
    )
    ticker_groups = ordered.groupby("ticker", sort=False)
    for column in feature_columns:
        limit = resolved_fill_limits.get(column)
        if limit is None:
            continue
        ordered[column] = ticker_groups[column].ffill(limit=limit)
    missing_after = int(ordered[feature_columns].isna().sum().sum())
    LOGGER.info(
        "Forward-filled feature columns within each ticker with family-specific limits: missing values %d -> %d.",
        missing_before,
        missing_after,
    )
    return ordered.sort_values(["date", "ticker"]).reset_index(drop=True)


def save_preprocessing_contract_artifacts(data: pd.DataFrame) -> None:
    save_preprocessing_contract_artifacts_from_columns(list(data.columns))


def save_preprocessing_contract_artifacts_from_columns(columns: list[str]) -> None:
    feature_registry = build_feature_registry_from_columns(columns)
    save_feature_registry(
        feature_registry,
        PREPROCESSED_FEATURE_REGISTRY_PARQUET,
        PREPROCESSED_FEATURE_REGISTRY_JSON,
    )
    feature_names = sorted(feature_registry[FEATURE_NAME_COLUMN].tolist())
    save_feature_schema_manifest(
        feature_names,
        PREPROCESSED_FEATURE_SCHEMA_MANIFEST_JSON,
    )
    LOGGER.info(
        "Saved preprocessing feature contract artifacts: features=%d",
        len(feature_names),
    )


def validate_no_missing_values(data: pd.DataFrame) -> None:
    total_missing: int = int(data.isna().sum().sum())
    if total_missing == 0:
        return

    missing_by_column: pd.Series = data.isna().sum()
    columns_with_missing: pd.Series = pd.Series(missing_by_column[missing_by_column > 0])
    details: str = ", ".join(
        f"{column}={int(count)}"
        for column, count in columns_with_missing.items()
    )
    raise ValueError(f"Missing values remain in preprocessed dataset: {details}")


def validate_required_columns_not_missing(
    data: pd.DataFrame,
    required_columns: list[str] | tuple[str, ...],
) -> None:
    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name in data.columns and data[column_name].isna().any()
    ]
    if not missing_columns:
        return

    details = ", ".join(
        f"{column_name}={int(data[column_name].isna().sum())}"
        for column_name in missing_columns
    )
    raise ValueError(f"Required columns still contain missing values: {details}")


@njit(cache=True, parallel=True)
def _distance_correlation_numba(x: np.ndarray, y: np.ndarray) -> float:
    n: int = x.shape[0]
    if n < 2:
        return np.nan

    a = np.empty((n, n), dtype=np.float64)
    b = np.empty((n, n), dtype=np.float64)
    a_row_mean = np.empty(n, dtype=np.float64)
    b_row_mean = np.empty(n, dtype=np.float64)
    a_col_mean = np.empty(n, dtype=np.float64)
    b_col_mean = np.empty(n, dtype=np.float64)

    a_total = 0.0
    b_total = 0.0

    for i in prange(n):
        row_sum_a = 0.0
        row_sum_b = 0.0
        xi = x[i]
        yi = y[i]
        for j in range(n):
            a_ij = abs(xi - x[j])
            b_ij = abs(yi - y[j])
            a[i, j] = a_ij
            b[i, j] = b_ij
            row_sum_a += a_ij
            row_sum_b += b_ij
        a_row_mean[i] = row_sum_a / n
        b_row_mean[i] = row_sum_b / n
        a_total += row_sum_a
        b_total += row_sum_b

    for j in prange(n):
        col_sum_a = 0.0
        col_sum_b = 0.0
        for i in range(n):
            col_sum_a += a[i, j]
            col_sum_b += b[i, j]
        a_col_mean[j] = col_sum_a / n
        b_col_mean[j] = col_sum_b / n

    a_total_mean = a_total / (n * n)
    b_total_mean = b_total / (n * n)

    dcov2 = 0.0
    dvar_x = 0.0
    dvar_y = 0.0
    for i in prange(n):
        for j in range(n):
            a_centered = a[i, j] - a_row_mean[i] - a_col_mean[j] + a_total_mean
            b_centered = b[i, j] - b_row_mean[i] - b_col_mean[j] + b_total_mean
            dcov2 += a_centered * b_centered
            dvar_x += a_centered * a_centered
            dvar_y += b_centered * b_centered

    dcov2 /= n * n
    dvar_x /= n * n
    dvar_y /= n * n

    if dvar_x <= 0.0 or dvar_y <= 0.0:
        return 0.0

    denom = np.sqrt(dvar_x * dvar_y)
    if denom <= 0.0:
        return 0.0

    ratio = dcov2 / denom
    if ratio < 0.0:
        ratio = 0.0
    return np.sqrt(ratio)


def _distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.ascontiguousarray(x.astype(np.float64))
    y_arr = np.ascontiguousarray(y.astype(np.float64))
    return float(_distance_correlation_numba(x_arr, y_arr))


def _get_candidate_feature_columns(
    data: pd.DataFrame,
    target_column: str,
    split_column: str,
) -> list[str]:
    excluded_columns: set[str] = {"date", "ticker", target_column, split_column}
    candidate_columns: list[str] = []
    for column in data.columns:
        if column in excluded_columns:
            continue
        if not pd.api.types.is_numeric_dtype(data[column]):
            continue
        if data[column].nunique(dropna=False) <= 1:
            continue
        candidate_columns.append(column)
    return candidate_columns


def _sample_train_subset(train_data: pd.DataFrame, sample_frac: float) -> pd.DataFrame:
    if train_data.empty:
        return train_data.copy()
    if sample_frac >= 1.0:
        sampled = train_data.copy()
    else:
        sampled = train_data.sample(frac=sample_frac, random_state=RANDOM_SEED)
    if len(sampled) > FEATURE_SAMPLE_MAX_ROWS:
        sampled = sampled.sample(n=FEATURE_SAMPLE_MAX_ROWS, random_state=RANDOM_SEED)
        LOGGER.info(
            "Capped correlated-feature pruning sample to %d train rows.",
            FEATURE_SAMPLE_MAX_ROWS,
        )
    return sampled.sort_values(["date", "ticker"]).reset_index(drop=True)


def _build_graph_adjacency(
    edges: list[tuple[str, str]],
    nodes: list[str],
) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def _collect_connected_component(
    start_node: str,
    adjacency: dict[str, set[str]],
    visited: set[str],
) -> list[str]:
    stack: list[str] = [start_node]
    component: list[str] = []
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        component.append(current)
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                stack.append(neighbor)
    return component


def _find_connected_components(edges: list[tuple[str, str]], nodes: list[str]) -> list[list[str]]:
    adjacency = _build_graph_adjacency(edges, nodes)
    components: list[list[str]] = []
    visited: set[str] = set()
    for node in nodes:
        if node in visited or not adjacency[node]:
            continue
        component = _collect_connected_component(node, adjacency, visited)
        if len(component) > 1:
            components.append(sorted(component))
    return components


def _build_abs_pearson_matrix(sampled_features: pd.DataFrame) -> pd.DataFrame:
    raw_corr = pd.DataFrame(sampled_features.corr(method="pearson"))
    return pd.DataFrame(
        np.abs(raw_corr.to_numpy()),
        index=sampled_features.columns,
        columns=sampled_features.columns,
    )


def _pearson_corr_value(
    pearson_corr: pd.DataFrame,
    left_column: str,
    right_column: str,
) -> float | None:
    corr_raw: Any = pearson_corr.at[left_column, right_column]
    if pd.isna(corr_raw):
        return None
    try:
        corr_scalar = np.asarray(corr_raw, dtype=np.float64).item()
        return float(corr_scalar)
    except (TypeError, ValueError):
        return None


def _build_candidate_pairs(
    candidate_columns: list[str],
    pearson_corr: pd.DataFrame,
    prescreener_threshold: float,
) -> list[tuple[str, str]]:
    candidate_pairs: list[tuple[str, str]] = []
    for left_index, left_column in enumerate(candidate_columns):
        for right_column in candidate_columns[left_index + 1:]:
            corr_value = _pearson_corr_value(pearson_corr, left_column, right_column)
            if corr_value is not None and corr_value >= prescreener_threshold:
                candidate_pairs.append((left_column, right_column))
    return candidate_pairs


def _distance_to_target(
    sampled_train: pd.DataFrame,
    column: str,
    target_values: np.ndarray,
) -> float:
    column_values = sampled_train[column].to_numpy(dtype=np.float64)
    return _distance_correlation(column_values, target_values)


def _evaluate_candidate_pairs(
    sampled_train: pd.DataFrame,
    candidate_pairs: list[tuple[str, str]],
    target_column: str,
    distance_threshold: float,
) -> tuple[list[tuple[str, str]], dict[str, float]]:
    target_values: np.ndarray = sampled_train[target_column].to_numpy(dtype=np.float64)
    target_scores: dict[str, float] = {}
    strong_edges: list[tuple[str, str]] = []
    progress_interval: int = max(1, len(candidate_pairs) // 20)

    for pair_index, (left_column, right_column) in enumerate(candidate_pairs, start=1):
        pair_score = _distance_correlation(
            sampled_train[left_column].to_numpy(dtype=np.float64),
            sampled_train[right_column].to_numpy(dtype=np.float64),
        )
        if pair_score >= distance_threshold:
            strong_edges.append((left_column, right_column))
            if left_column not in target_scores:
                target_scores[left_column] = _distance_to_target(
                    sampled_train,
                    left_column,
                    target_values,
                )
            if right_column not in target_scores:
                target_scores[right_column] = _distance_to_target(
                    sampled_train,
                    right_column,
                    target_values,
                )
        if pair_index == 1 or pair_index % progress_interval == 0 or pair_index == len(candidate_pairs):
            LOGGER.info(
                "Distance-correlation pruning progress: %d/%d pairs evaluated, %d strong pairs found.",
                pair_index,
                len(candidate_pairs),
                len(strong_edges),
            )
    return strong_edges, target_scores


def _select_columns_to_drop(
    strong_edges: list[tuple[str, str]],
    candidate_columns: list[str],
    target_scores: dict[str, float],
) -> set[str]:
    components = _find_connected_components(strong_edges, candidate_columns)
    columns_to_drop: set[str] = set()
    for component in components:
        keep_column = max(
            component,
            key=lambda column: (target_scores.get(column, 0.0), column),
        )
        for column in component:
            if column != keep_column:
                columns_to_drop.add(column)
    return columns_to_drop


def prune_correlated_features(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    split_column: str = SPLIT_COLUMN,
    feature_sample_frac: float = FEATURE_SAMPLE_FRAC,
    prescreener_threshold: float = PEARSON_PRESCREENER_THRESHOLD,
    distance_threshold: float = DISTANCE_CORRELATION_THRESHOLD,
) -> pd.DataFrame:
    train_data: pd.DataFrame = pd.DataFrame(data.loc[data[split_column] == "train"].copy())
    sampled_train: pd.DataFrame = _sample_train_subset(train_data, feature_sample_frac)
    candidate_columns: list[str] = _get_candidate_feature_columns(
        sampled_train,
        target_column,
        split_column,
    )
    LOGGER.info(
        "Starting correlated-feature pruning on %d sampled train rows across %d candidate features.",
        len(sampled_train),
        len(candidate_columns),
    )
    if not NUMBA_AVAILABLE:
        LOGGER.warning(
            "Numba is not installed in the active environment. "
            "Distance-correlation pruning is running in a slow single-core fallback mode. "
            "Install project dependencies to enable the optimized path.",
        )
    if len(candidate_columns) < 2:
        return data

    sampled_features: pd.DataFrame = pd.DataFrame(sampled_train.loc[:, candidate_columns])
    pearson_corr = _build_abs_pearson_matrix(sampled_features)
    candidate_pairs = _build_candidate_pairs(
        candidate_columns,
        pearson_corr,
        prescreener_threshold,
    )

    LOGGER.info(
        "Pearson prescreener retained %d candidate feature pairs above %.2f.",
        len(candidate_pairs),
        prescreener_threshold,
    )

    if not candidate_pairs:
        return data

    strong_edges, target_scores = _evaluate_candidate_pairs(
        sampled_train,
        candidate_pairs,
        target_column,
        distance_threshold,
    )

    if not strong_edges:
        return data

    columns_to_drop = _select_columns_to_drop(
        strong_edges,
        candidate_columns,
        target_scores,
    )

    if not columns_to_drop:
        return data

    pruned: pd.DataFrame = data.drop(columns=sorted(columns_to_drop))
    LOGGER.info(
        "Pruned %d correlated features using distance correlation on train data.",
        len(columns_to_drop),
    )
    return pruned


def save_preprocessed_dataset(
    data: pd.DataFrame,
    parquet_path: Path,
    csv_path: Path,
) -> dict[str, Path]:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    data.to_parquet(parquet_path, index=False)
    sample: pd.DataFrame = data.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)
    sample = sample.sort_values(["date", "ticker"]).reset_index(drop=True)
    sample.to_csv(csv_path, index=False)
    LOGGER.info(
        "Saved preprocessed parquet: %s (%d rows x %d cols)",
        parquet_path,
        len(data),
        len(data.columns),
    )
    LOGGER.info("Saved preprocessed sample CSV: %s", csv_path)
    return {"parquet": parquet_path, "sample_csv": csv_path}


def save_research_label_panel(data: pd.DataFrame) -> None:
    label_columns = ["date", "ticker", *TARGET_RELATED_COLUMNS]
    label_panel = pd.DataFrame(data.loc[:, [column for column in label_columns if column in data.columns]].copy())
    PREPROCESSED_RESEARCH_LABEL_PANEL_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    label_panel.to_parquet(PREPROCESSED_RESEARCH_LABEL_PANEL_PARQUET, index=False)
    LOGGER.info(
        "Saved research label panel: %s (%d rows x %d cols)",
        PREPROCESSED_RESEARCH_LABEL_PANEL_PARQUET,
        len(label_panel),
        len(label_panel.columns),
    )


def build_protected_columns() -> list[str]:
    return ["date", "ticker", SPLIT_COLUMN, *TARGET_RELATED_COLUMNS]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    DATA_PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    from core.src.meta_model.data.data_preprocessing.streaming import run_streaming_preprocessing

    run_streaming_preprocessing(
        FEATURES_OUTPUT_PARQUET,
        output_parquet_path=PREPROCESSED_OUTPUT_PARQUET,
        output_csv_path=PREPROCESSED_OUTPUT_SAMPLE_CSV,
        train_parquet_path=PREPROCESSED_TRAIN_PARQUET,
        train_csv_path=PREPROCESSED_TRAIN_SAMPLE_CSV,
        val_parquet_path=PREPROCESSED_VAL_PARQUET,
        val_csv_path=PREPROCESSED_VAL_SAMPLE_CSV,
        test_parquet_path=PREPROCESSED_TEST_PARQUET,
        test_csv_path=PREPROCESSED_TEST_SAMPLE_CSV,
        label_panel_path=PREPROCESSED_RESEARCH_LABEL_PANEL_PARQUET,
    )
    LOGGER.info("Data preprocessing pipeline completed.")


__all__ = [
    "MODEL_TARGET_COLUMN",
    "TARGET_COLUMN",
    "TARGET_RELATED_COLUMNS",
    "SPLIT_COLUMN",
    "assign_dataset_splits",
    "build_protected_columns",
    "create_target_main",
    "drop_fully_missing_feature_columns",
    "exclude_covid_period",
    "drop_columns_with_missing_values",
    "filter_from_start_date",
    "build_feature_fill_limits",
    "forward_fill_features_by_ticker",
    "load_feature_dataset",
    "main",
    "prune_correlated_features",
    "remove_rows_with_missing_values",
    "save_preprocessed_dataset",
    "validate_required_columns_not_missing",
    "validate_no_missing_values",
]


if __name__ == "__main__":
    main()
