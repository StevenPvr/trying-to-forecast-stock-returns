from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from core.src.meta_model.data.constants import RANDOM_SEED, SAMPLE_FRAC
from core.src.meta_model.data.paths import DATA_FEATURE_CORR_PCA_DIR
from core.src.meta_model.feature_corr_pca.config import TRANSFORM_BATCH_ROWS, resolve_max_workers
from core.src.meta_model.feature_corr_pca.dataset_utils import iter_dataset_batches, record_batch_to_frame
from core.src.meta_model.feature_corr_pca.models import KernelPCAGroupModel

LOGGER: logging.Logger = logging.getLogger(__name__)
TRANSFORM_PROGRESS_EVERY_BATCHES: int = 25


def write_sample_chunk(
    sample_frame: pd.DataFrame,
    output_sample_csv_path: Path,
    write_header: bool,
) -> bool:
    if sample_frame.empty:
        return write_header
    sample_frame.to_csv(
        output_sample_csv_path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
    )
    return False


def stream_transform_to_output(
    feature_parquet_path: Path,
    output_parquet_path: Path,
    output_sample_csv_path: Path,
    columns_to_drop: set[str],
    models: list[KernelPCAGroupModel],
    transform_feature_fn: Callable[[pd.DataFrame, KernelPCAGroupModel], np.ndarray],
    start_date: date | pd.Timestamp | None = None,
    max_workers: int | None = None,
) -> tuple[int, int]:
    if output_parquet_path.exists():
        output_parquet_path.unlink()
    if output_sample_csv_path.exists():
        output_sample_csv_path.unlink()

    sample_header_needed = True
    writer: pq.ParquetWriter | None = None
    rng = np.random.default_rng(RANDOM_SEED)
    row_count = 0
    column_count = 0
    last_output_columns: list[str] = []
    worker_count = min(resolve_max_workers(max_workers), max(1, len(models)))
    all_columns = list(pq.read_schema(feature_parquet_path).names)
    stage_started_at = time.perf_counter()
    processed_batches = 0
    LOGGER.info(
        "Starting transform/output stage: %d component models | batch_rows=%d | workers=%d.",
        len(models),
        TRANSFORM_BATCH_ROWS,
        worker_count,
    )

    executor: ThreadPoolExecutor | None = None
    if models and worker_count > 1:
        executor = ThreadPoolExecutor(max_workers=worker_count)

    try:
        for batch in iter_dataset_batches(
            feature_parquet_path,
            columns=all_columns,
            batch_size=TRANSFORM_BATCH_ROWS,
            train_only=False,
            start_date=start_date,
            use_threads=False,
        ):
            batch_frame = record_batch_to_frame(batch)
            output_frame = batch_frame.drop(columns=sorted(columns_to_drop), errors="ignore")

            if models:
                if executor is None:
                    transformed_columns = [
                        (model.component_feature_name, transform_feature_fn(batch_frame, model))
                        for model in models
                    ]
                else:
                    futures = [
                        (model.component_feature_name, executor.submit(transform_feature_fn, batch_frame, model))
                        for model in models
                    ]
                    transformed_columns = [
                        (column_name, future.result())
                        for column_name, future in futures
                    ]

                transformed_frame = pd.DataFrame(
                    {
                        column_name: values
                        for column_name, values in transformed_columns
                    },
                    index=output_frame.index,
                )
                output_frame = pd.concat([output_frame, transformed_frame], axis=1)

            last_output_columns = output_frame.columns.tolist()
            output_table = pa.Table.from_pandas(output_frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_parquet_path, output_table.schema)
                column_count = len(output_frame.columns)
            writer.write_table(output_table)
            row_count += len(output_frame)
            processed_batches += 1

            sampled_mask = rng.random(len(output_frame)) < SAMPLE_FRAC
            sampled_rows = output_frame.loc[sampled_mask].reset_index(drop=True)
            sample_header_needed = write_sample_chunk(
                sampled_rows,
                output_sample_csv_path,
                write_header=sample_header_needed,
            )
            if processed_batches == 1 or processed_batches % TRANSFORM_PROGRESS_EVERY_BATCHES == 0:
                LOGGER.info(
                    "Transform/output stage progress: %d batches written | %d rows materialized | elapsed=%.2fs",
                    processed_batches,
                    row_count,
                    time.perf_counter() - stage_started_at,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if writer is not None:
            writer.close()

    if sample_header_needed and last_output_columns:
        pd.DataFrame(columns=last_output_columns).to_csv(output_sample_csv_path, index=False)
    LOGGER.info(
        "Transform/output stage completed: %d rows x %d cols | batches=%d | elapsed=%.2fs.",
        row_count,
        column_count,
        processed_batches,
        time.perf_counter() - stage_started_at,
    )
    return row_count, column_count


def save_feature_corr_pca_outputs(
    data: pd.DataFrame,
    mapping: dict[str, object],
    output_parquet_path: Path,
    output_sample_csv_path: Path,
    output_mapping_json_path: Path,
) -> None:
    DATA_FEATURE_CORR_PCA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_parquet_path, index=False)
    sample = (
        data.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    sample.to_csv(output_sample_csv_path, index=False)
    output_mapping_json_path.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    LOGGER.info(
        "Saved feature corr+pca parquet: %s (%d rows x %d cols)",
        output_parquet_path,
        len(data),
        len(data.columns),
    )
    LOGGER.info("Saved feature corr+pca mapping JSON: %s", output_mapping_json_path)
