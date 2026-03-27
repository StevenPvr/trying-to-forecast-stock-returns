from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.src.meta_model.data.paths import PREPROCESSED_OUTPUT_PARQUET
from core.src.meta_model.evaluate.config import (
    DATE_COLUMN,
    EXCLUDED_FEATURE_COLUMNS,
    SPLIT_COLUMN,
    TEST_SPLIT_NAME,
    TICKER_COLUMN,
    TRAIN_SPLIT_NAME,
    VAL_SPLIT_NAME,
)


def load_preprocessed_evaluation_dataset(
    path: Path = PREPROCESSED_OUTPUT_PARQUET,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    data = pd.read_parquet(path)
    prepared = data.copy()
    prepared[DATE_COLUMN] = pd.to_datetime(prepared[DATE_COLUMN])
    return prepared.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)


def build_feature_columns(data: pd.DataFrame) -> list[str]:
    return sorted(
        column_name
        for column_name in data.columns
        if column_name not in EXCLUDED_FEATURE_COLUMNS
    )


def split_training_and_test_frames(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = data[SPLIT_COLUMN].isin([TRAIN_SPLIT_NAME, VAL_SPLIT_NAME])
    test_mask = data[SPLIT_COLUMN] == TEST_SPLIT_NAME
    training_frame = pd.DataFrame(data.loc[train_mask].copy())
    test_frame = pd.DataFrame(data.loc[test_mask].copy())
    if training_frame.empty:
        raise ValueError("No training rows found for evaluate pipeline.")
    if test_frame.empty:
        raise ValueError("No test rows found for evaluate pipeline.")
    return (
        training_frame.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True),
        test_frame.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True),
    )
