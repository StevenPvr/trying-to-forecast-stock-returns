from __future__ import annotations

"""Data-cleaning orchestrator: load raw dataset, apply outlier pipeline, save cleaned output."""

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.constants import RANDOM_SEED, SAMPLE_FRAC
from core.src.meta_model.data.data_cleaning.outlier_pipeline import apply_outlier_flags
from core.src.meta_model.data.data_cleaning.outlier_plots import create_outlier_plots
from core.src.meta_model.data.paths import (
    CLEANED_OUTPUT_PARQUET,
    CLEANED_OUTPUT_SAMPLE_CSV,
    MERGED_OUTPUT_PARQUET,
    OUTLIER_PLOTS_DIR,
)

LOGGER: logging.Logger = logging.getLogger(__name__)

_QUASI_CONSTANT_SHARE_THRESHOLD: float = 0.99


def load_raw_dataset(path: Path) -> pd.DataFrame:
    """Load the merged parquet produced by data_fetching."""
    if not path.exists():
        raise FileNotFoundError(f"Input parquet not found: {path}")
    df: pd.DataFrame = pd.read_parquet(path)
    LOGGER.info("Loaded raw dataset: %d rows x %d cols", len(df), len(df.columns))
    return df


def log_nan_report(df: pd.DataFrame, stage: str) -> None:
    """Log NaN percentage per column at a given pipeline stage."""
    total_nan: int = int(df.isna().sum().sum())
    n_rows: int = len(df)
    LOGGER.info("[%s] Total NaN: %d", stage, total_nan)
    if total_nan == 0 or n_rows == 0:
        return
    nan_per_col: pd.Series = df.isna().sum()
    cols_with_nan: pd.Series = nan_per_col[nan_per_col > 0]
    for col, count in cols_with_nan.items():
        pct: float = 100.0 * count / n_rows
        LOGGER.info("[%s]   %s: %d NaN (%.1f%%)", stage, col, count, pct)


def log_nan_report_by_ticker(df: pd.DataFrame, stage: str) -> None:
    """Log overall NaN percentage per ticker, sorted worst-first."""
    if "ticker" not in df.columns:
        LOGGER.info("[%s] No 'ticker' column — skipping per-ticker NaN report.", stage)
        return
    numeric_cols: list[str] = [
        c for c in df.columns if c not in ("date", "ticker")
    ]
    if not numeric_cols:
        return
    nan_by_ticker: pd.Series = df.groupby("ticker")[numeric_cols].apply(
        lambda g: 100.0 * g.isna().sum().sum() / g.size,
    )
    nan_by_ticker = nan_by_ticker[nan_by_ticker > 0].sort_values(ascending=False)
    LOGGER.info(
        "[%s] %d / %d tickers with missing data:",
        stage, len(nan_by_ticker), df["ticker"].nunique(),
    )
    for ticker, pct in nan_by_ticker.items():
        LOGGER.info("[%s]   %s: %.2f%%", stage, ticker, pct)


def save_cleaned(
    data: pd.DataFrame,
    parquet_path: Path,
    csv_path: Path,
) -> dict[str, Path]:
    """Save cleaned parquet + 5% sample CSV."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(parquet_path, index=False)
    sample: pd.DataFrame = data.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)
    sample = sample.sort_values(["date", "ticker"]).reset_index(drop=True)
    sample.to_csv(csv_path, index=False)
    LOGGER.info(
        "Saved cleaned parquet: %s (%d rows x %d cols)",
        parquet_path, len(data), len(data.columns),
    )
    LOGGER.info("Saved cleaned sample CSV: %s", csv_path)
    return {"parquet": parquet_path, "sample_csv": csv_path}


def _is_binary_indicator(series: pd.Series) -> bool:
    if not (
        pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_numeric_dtype(series)
    ):
        return False
    non_null_values = pd.Series(series.dropna().unique())
    if non_null_values.empty or len(non_null_values) > 2:
        return False
    return bool(non_null_values.isin([0, 1, 0.0, 1.0, False, True]).all())


def _find_quasi_constant_columns(
    df: pd.DataFrame,
    candidate_columns: list[str],
    threshold: float = _QUASI_CONSTANT_SHARE_THRESHOLD,
) -> list[str]:
    quasi_constant_columns: list[str] = []
    for col in candidate_columns:
        series: pd.Series = df[col]
        if _is_binary_indicator(series):
            continue
        value_counts: pd.Series = series.value_counts(dropna=False)
        if value_counts.empty:
            continue
        dominant_share: float = float(value_counts.iloc[0]) / float(len(series))
        if dominant_share >= threshold:
            quasi_constant_columns.append(col)
    return quasi_constant_columns


def finalize_modeling_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the final cleaned dataset for modeling.

    - Exclude rows flagged as data errors.
    - Keep only quantitative outlier indicators as binary features.
    - Drop diagnostic and redundant outlier columns.
    - Drop constant non-identifier columns.
    """
    finalized: pd.DataFrame = df.copy()

    if "data_error_flag" in finalized.columns:
        initial_rows: int = len(finalized)
        error_flag_series: pd.Series = finalized["data_error_flag"]
        filtered: pd.DataFrame = finalized.loc[
            ~error_flag_series.astype(bool)
        ].copy()
        finalized = filtered
        dropped_rows: int = initial_rows - len(finalized)
        LOGGER.info(
            "Filtered out %d rows flagged as data errors (%d -> %d).",
            dropped_rows,
            initial_rows,
            len(finalized),
        )

    binary_feature_columns: tuple[str, ...] = (
        "ticker_return_extreme_flag",
        "cross_section_return_extreme_flag",
    )
    for col in binary_feature_columns:
        if col in finalized.columns:
            finalized[col] = finalized[col].astype("int8")

    columns_to_drop: list[str] = [
        col
        for col in (
            "data_error_flag",
            "is_outlier_flag",
            "outlier_severity",
            "outlier_reason",
        )
        if col in finalized.columns
    ]
    if columns_to_drop:
        dropped_outlier_columns: pd.DataFrame = finalized.drop(columns=columns_to_drop)
        finalized = dropped_outlier_columns
        LOGGER.info(
            "Dropped non-modeling outlier columns: %s",
            ", ".join(columns_to_drop),
        )

    identifier_columns: set[str] = {"date", "ticker"}
    candidate_columns: list[str] = [
        col for col in finalized.columns if col not in identifier_columns
    ]
    constant_columns: list[str] = [
        col
        for col in candidate_columns
        if finalized[col].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        dropped_constant_columns: pd.DataFrame = finalized.drop(columns=constant_columns)
        finalized = dropped_constant_columns
        LOGGER.info(
            "Dropped constant modeling columns: %s",
            ", ".join(constant_columns),
        )
        candidate_columns = [
            col for col in candidate_columns if col not in constant_columns
        ]

    quasi_constant_columns: list[str] = _find_quasi_constant_columns(
        finalized,
        candidate_columns,
    )
    if quasi_constant_columns:
        dropped_quasi_constant_columns: pd.DataFrame = finalized.drop(
            columns=quasi_constant_columns,
        )
        finalized = dropped_quasi_constant_columns
        LOGGER.info(
            "Dropped quasi-constant modeling columns (>= %.3f dominant share): %s",
            _QUASI_CONSTANT_SHARE_THRESHOLD,
            ", ".join(quasi_constant_columns),
        )

    return finalized.reset_index(drop=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    df: pd.DataFrame = load_raw_dataset(MERGED_OUTPUT_PARQUET)
    log_nan_report(df, "before_cleaning")
    log_nan_report_by_ticker(df, "before_cleaning")
    cleaned_with_diagnostics: pd.DataFrame = apply_outlier_flags(df)
    log_nan_report(cleaned_with_diagnostics, "after_cleaning")
    log_nan_report_by_ticker(cleaned_with_diagnostics, "after_cleaning")
    create_outlier_plots(cleaned_with_diagnostics, OUTLIER_PLOTS_DIR)

    modeling_ready: pd.DataFrame = finalize_modeling_dataset(
        cleaned_with_diagnostics,
    )
    save_cleaned(modeling_ready, CLEANED_OUTPUT_PARQUET, CLEANED_OUTPUT_SAMPLE_CSV)
    LOGGER.info("Data cleaning pipeline completed.")


if __name__ == "__main__":
    main()
