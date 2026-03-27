from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from core.src.meta_model.data.paths import FEATURES_OUTPUT_PARQUET
from core.src.meta_model.feature_corr_pca.config import NON_FEATURE_COLUMNS, TRAIN_END_DATE

LOGGER: logging.Logger = logging.getLogger(__name__)


def train_filter_expression(train_end_date: date = TRAIN_END_DATE) -> ds.Expression:
    return ds.field("date") <= pd.Timestamp(train_end_date)


def start_filter_expression(start_date: date | pd.Timestamp) -> ds.Expression:
    return ds.field("date") >= pd.Timestamp(start_date)


def build_dataset_filter(
    *,
    train_only: bool = False,
    train_end_date: date = TRAIN_END_DATE,
    start_date: date | pd.Timestamp | None = None,
) -> ds.Expression | None:
    filter_expression: ds.Expression | None = None
    if start_date is not None:
        filter_expression = start_filter_expression(start_date)
    if train_only:
        train_expression = train_filter_expression(train_end_date)
        filter_expression = (
            train_expression
            if filter_expression is None
            else filter_expression & train_expression
        )
    return filter_expression


def iter_dataset_batches(
    path: Path,
    columns: list[str],
    batch_size: int,
    train_only: bool = False,
    train_end_date: date = TRAIN_END_DATE,
    start_date: date | pd.Timestamp | None = None,
    use_threads: bool = True,
) -> Any:
    dataset = ds.dataset(path, format="parquet")
    scanner = dataset.scanner(
        columns=columns,
        filter=build_dataset_filter(
            train_only=train_only,
            train_end_date=train_end_date,
            start_date=start_date,
        ),
        batch_size=batch_size,
        use_threads=use_threads,
    )
    return scanner.to_batches()


def record_batch_to_frame(batch: pa.RecordBatch) -> pd.DataFrame:
    return batch.to_pandas(split_blocks=True, self_destruct=True)


def load_feature_dataset(path: Path = FEATURES_OUTPUT_PARQUET) -> pd.DataFrame:
    data = pd.read_parquet(path)
    LOGGER.info(
        "Loaded feature dataset for corr+pca: %d rows x %d cols",
        len(data),
        len(data.columns),
    )
    return data


def build_candidate_feature_columns(columns: list[str] | pd.Index) -> list[str]:
    return [column for column in columns if column not in NON_FEATURE_COLUMNS]


def select_train_frame_for_correlation(
    data: pd.DataFrame,
    train_end_date: date = TRAIN_END_DATE,
) -> pd.DataFrame:
    dated = data.copy()
    dated["date"] = pd.to_datetime(dated["date"])
    train_frame = dated.loc[dated["date"] <= pd.Timestamp(train_end_date)].reset_index(drop=True)
    LOGGER.info(
        "Selected train-only frame for corr+pca fit: %d rows x %d cols through %s.",
        len(train_frame),
        len(train_frame.columns),
        pd.Timestamp(train_end_date).date(),
    )
    return train_frame
