from __future__ import annotations

import logging
import importlib
import importlib.util
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_numba_spec = importlib.util.find_spec("numba")
NUMBA_AVAILABLE: bool = _numba_spec is not None
if _numba_spec is not None:  # pragma: no branch
    _numba = importlib.import_module("numba")
    njit = _numba.njit
    prange = _numba.prange
else:  # pragma: no cover - exercised when numba is unavailable locally
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    def prange(*args):
        return range(*args)

from core.src.meta_model.data.constants import RANDOM_SEED, SAMPLE_FRAC
from core.src.meta_model.data.paths import (
    DATA_PREPROCESSING_DIR,
    GREEDY_FORWARD_SELECTION_FILTERED_FEATURES_PARQUET,
    PREPROCESSED_OUTPUT_PARQUET,
    PREPROCESSED_OUTPUT_SAMPLE_CSV,
    PREPROCESSED_TEST_PARQUET,
    PREPROCESSED_TEST_SAMPLE_CSV,
    PREPROCESSED_TRAIN_PARQUET,
    PREPROCESSED_TRAIN_SAMPLE_CSV,
    PREPROCESSED_VAL_PARQUET,
    PREPROCESSED_VAL_SAMPLE_CSV,
)

LOGGER: logging.Logger = logging.getLogger(__name__)

PREPROCESSING_START_DATE: date = date(2009, 1, 1)
COVID_START_DATE: date = date(2020, 2, 1)
COVID_END_DATE: date = date(2021, 12, 31)
DateLike = str | date | datetime
TARGET_COLUMN: str = "target_main"
SPLIT_COLUMN: str = "dataset_split"
TARGET_HORIZON_DAYS: int = 5
FEATURE_SAMPLE_FRAC: float = 0.5
FEATURE_SAMPLE_MAX_ROWS: int = 2000
PEARSON_PRESCREENER_THRESHOLD: float = 0.9
DISTANCE_CORRELATION_THRESHOLD: float = 0.95
TRAIN_END_DATE: date = date(2018, 11, 30)
VAL_START_DATE: date = date(2019, 2, 1)
VAL_END_DATE: date = date(2021, 11, 30)
TEST_START_DATE: date = date(2022, 2, 1)


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


def create_target_main(
    data: pd.DataFrame,
    horizon_days: int = TARGET_HORIZON_DAYS,
) -> pd.DataFrame:
    enriched: pd.DataFrame = data.sort_values(["ticker", "date"]).reset_index(drop=True).copy()

    if "stock_close_price" not in enriched.columns:
        raise ValueError("Missing required column for target creation: stock_close_price")

    target_parts: list[pd.DataFrame] = []
    for _, group in enriched.groupby("ticker", sort=False):
        ticker_group: pd.DataFrame = group.copy()
        future_close: pd.Series = pd.Series(
            ticker_group["stock_close_price"].shift(-horizon_days),
        )
        ticker_group[TARGET_COLUMN] = np.log(future_close / ticker_group["stock_close_price"])
        target_parts.append(ticker_group)

    result: pd.DataFrame = pd.concat(target_parts, ignore_index=True)
    LOGGER.info(
        "Created %s using a %d-trading-day forward close log return horizon.",
        TARGET_COLUMN,
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
    bridge_dates_before_covid = set(pre_covid_dates[-target_horizon_days:].tolist())
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
    split_ready: pd.DataFrame = data.copy()
    dates: pd.Series = pd.to_datetime(split_ready["date"])
    train_end_timestamp = pd.Timestamp(TRAIN_END_DATE)
    val_start_timestamp = pd.Timestamp(VAL_START_DATE)
    val_end_timestamp = pd.Timestamp(VAL_END_DATE)
    test_start_timestamp = pd.Timestamp(TEST_START_DATE)
    split_ready[SPLIT_COLUMN] = pd.Series(pd.NA, index=split_ready.index, dtype="object")

    split_ready.loc[dates <= train_end_timestamp, SPLIT_COLUMN] = "train"
    split_ready.loc[
        (dates >= val_start_timestamp) & (dates <= val_end_timestamp),
        SPLIT_COLUMN,
    ] = "val"
    split_ready.loc[dates >= test_start_timestamp, SPLIT_COLUMN] = "test"

    split_ready = pd.DataFrame(split_ready.loc[split_ready[SPLIT_COLUMN].notna()].copy())
    LOGGER.info(
        "Assigned dataset splits: train=%d, val=%d, test=%d",
        int((split_ready[SPLIT_COLUMN] == "train").sum()),
        int((split_ready[SPLIT_COLUMN] == "val").sum()),
        int((split_ready[SPLIT_COLUMN] == "test").sum()),
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


def forward_fill_features_by_ticker(
    data: pd.DataFrame,
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
    ordered[feature_columns] = ordered.groupby("ticker", sort=False)[feature_columns].ffill()
    missing_after = int(ordered[feature_columns].isna().sum().sum())
    LOGGER.info(
        "Forward-filled feature columns within each ticker: missing values %d -> %d.",
        missing_before,
        missing_after,
    )
    return ordered.sort_values(["date", "ticker"]).reset_index(drop=True)


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


def _find_connected_components(edges: list[tuple[str, str]], nodes: list[str]) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    components: list[list[str]] = []
    visited: set[str] = set()
    for node in nodes:
        if node in visited or not adjacency[node]:
            continue
        stack: list[str] = [node]
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
        if len(component) > 1:
            components.append(sorted(component))
    return components


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
    raw_corr = pd.DataFrame(sampled_features.corr(method="pearson"))
    pearson_corr: pd.DataFrame = pd.DataFrame(
        np.abs(raw_corr.to_numpy()),
        index=candidate_columns,
        columns=candidate_columns,
    )
    candidate_pairs: list[tuple[str, str]] = []
    for left_index, left_column in enumerate(candidate_columns):
        for right_column in candidate_columns[left_index + 1:]:
            corr_value = pearson_corr.loc[left_column, right_column]
            if pd.notna(corr_value) and float(corr_value) >= prescreener_threshold:
                candidate_pairs.append((left_column, right_column))

    LOGGER.info(
        "Pearson prescreener retained %d candidate feature pairs above %.2f.",
        len(candidate_pairs),
        prescreener_threshold,
    )

    if not candidate_pairs:
        return data

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
                target_scores[left_column] = _distance_correlation(
                    sampled_train[left_column].to_numpy(dtype=np.float64),
                    target_values,
                )
            if right_column not in target_scores:
                target_scores[right_column] = _distance_correlation(
                    sampled_train[right_column].to_numpy(dtype=np.float64),
                    target_values,
                )
        if pair_index == 1 or pair_index % progress_interval == 0 or pair_index == len(candidate_pairs):
            LOGGER.info(
                "Distance-correlation pruning progress: %d/%d pairs evaluated, %d strong pairs found.",
                pair_index,
                len(candidate_pairs),
                len(strong_edges),
            )

    if not strong_edges:
        return data

    components: list[list[str]] = _find_connected_components(strong_edges, candidate_columns)
    columns_to_drop: set[str] = set()
    for component in components:
        keep_column = max(
            component,
            key=lambda column: (target_scores.get(column, 0.0), column),
        )
        for column in component:
            if column != keep_column:
                columns_to_drop.add(column)

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


def _save_split_datasets(data: pd.DataFrame) -> None:
    split_outputs: tuple[tuple[str, Path, Path], ...] = (
        ("train", PREPROCESSED_TRAIN_PARQUET, PREPROCESSED_TRAIN_SAMPLE_CSV),
        ("val", PREPROCESSED_VAL_PARQUET, PREPROCESSED_VAL_SAMPLE_CSV),
        ("test", PREPROCESSED_TEST_PARQUET, PREPROCESSED_TEST_SAMPLE_CSV),
    )
    for split_name, parquet_path, csv_path in split_outputs:
        split_df: pd.DataFrame = pd.DataFrame(
            data.loc[data[SPLIT_COLUMN] == split_name].copy(),
        )
        save_preprocessed_dataset(split_df, parquet_path, csv_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    DATA_PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    featured: pd.DataFrame = load_feature_dataset(GREEDY_FORWARD_SELECTION_FILTERED_FEATURES_PARQUET)
    filtered: pd.DataFrame = filter_from_start_date(featured)
    targeted: pd.DataFrame = create_target_main(filtered)
    covid_excluded: pd.DataFrame = exclude_covid_period(targeted)
    split_ready: pd.DataFrame = assign_dataset_splits(covid_excluded)
    forward_filled: pd.DataFrame = forward_fill_features_by_ticker(
        split_ready,
        protected_columns=["date", "ticker", TARGET_COLUMN, SPLIT_COLUMN],
    )
    rows_ready: pd.DataFrame = remove_rows_with_missing_values(
        forward_filled,
    )
    columns_ready: pd.DataFrame = drop_columns_with_missing_values(
        rows_ready,
        protected_columns=["date", "ticker", TARGET_COLUMN, SPLIT_COLUMN],
    )
    pruned: pd.DataFrame = prune_correlated_features(columns_ready)
    validate_no_missing_values(pruned)
    save_preprocessed_dataset(
        pruned,
        PREPROCESSED_OUTPUT_PARQUET,
        PREPROCESSED_OUTPUT_SAMPLE_CSV,
    )
    _save_split_datasets(pruned)
    LOGGER.info("Data preprocessing pipeline completed.")


__all__ = [
    "TARGET_COLUMN",
    "SPLIT_COLUMN",
    "assign_dataset_splits",
    "create_target_main",
    "exclude_covid_period",
    "drop_columns_with_missing_values",
    "filter_from_start_date",
    "forward_fill_features_by_ticker",
    "load_feature_dataset",
    "main",
    "prune_correlated_features",
    "remove_rows_with_missing_values",
    "save_preprocessed_dataset",
    "validate_no_missing_values",
]


if __name__ == "__main__":
    main()
