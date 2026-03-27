from __future__ import annotations

import json
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas._libs.tslibs.nattype import NaTType

PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.paths import (
    DATA_FEATURE_CORR_PCA_DIR,
    FEATURE_CORR_PCA_MAPPING_JSON,
    FEATURE_CORR_PCA_OUTPUT_PARQUET,
    FEATURE_CORR_PCA_OUTPUT_SAMPLE_CSV,
    FEATURES_OUTPUT_PARQUET,
)
from core.src.meta_model.feature_corr_pca.config import (
    DEFAULT_COMPONENTS_PER_GROUP,
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_KERNEL,
    TRAIN_END_DATE,
    resolve_max_workers,
)
from core.src.meta_model.feature_corr_pca.correlation import (
    find_correlated_feature_groups,
    find_correlated_feature_groups_from_parquet,
)
from core.src.meta_model.feature_corr_pca.dataset_utils import build_candidate_feature_columns, load_feature_dataset
from core.src.meta_model.feature_corr_pca.io import save_feature_corr_pca_outputs, stream_transform_to_output
from core.src.meta_model.feature_corr_pca.kernel_pca import (
    apply_feature_corr_kernel_pca,
    fit_group_models_from_parquet,
    transform_group_batch,
)

LOGGER: logging.Logger = logging.getLogger(__name__)
SECONDARY_PREDICTION_COLUMN_PREFIX: str = "pred_future_"


def build_secondary_prediction_feature_columns(columns: list[str] | pd.Index) -> list[str]:
    return [
        str(column_name)
        for column_name in columns
        if str(column_name).startswith(SECONDARY_PREDICTION_COLUMN_PREFIX)
    ]


def derive_secondary_prediction_start_date(
    feature_parquet_path: Path,
    *,
    available_columns: list[str] | pd.Index | None = None,
) -> pd.Timestamp | None:
    schema_columns = (
        list(available_columns)
        if available_columns is not None
        else list(pq.read_schema(feature_parquet_path).names)
    )
    prediction_columns = build_secondary_prediction_feature_columns(schema_columns)
    if not prediction_columns:
        return None

    availability_frame = pd.read_parquet(
        feature_parquet_path,
        columns=["date", *prediction_columns],
    )
    availability_frame["date"] = pd.to_datetime(availability_frame["date"])
    fully_available_dates = availability_frame.loc[
        availability_frame[prediction_columns].notna().all(axis=1),
        "date",
    ]
    if fully_available_dates.empty:
        raise ValueError(
            "Secondary prediction columns are present in the feature dataset but never fully available on any row."
        )

    start_date_raw = fully_available_dates.min()
    if isinstance(start_date_raw, NaTType) or pd.isna(start_date_raw):
        raise ValueError("Secondary prediction start date cannot be NaT.")
    start_date = pd.Timestamp(start_date_raw)
    if isinstance(start_date, NaTType) or pd.isna(start_date):
        raise ValueError("Secondary prediction start date cannot be NaT after normalization.")
    dropped_rows = int((availability_frame["date"] < start_date).sum())
    LOGGER.info(
        "Detected secondary-model burn-in: dropping %d rows before %s for meta-model corr+pca.",
        dropped_rows,
        start_date.date(),
    )
    return start_date


def run_feature_corr_pca(
    feature_parquet_path: Path = FEATURES_OUTPUT_PARQUET,
    output_parquet_path: Path = FEATURE_CORR_PCA_OUTPUT_PARQUET,
    output_sample_csv_path: Path = FEATURE_CORR_PCA_OUTPUT_SAMPLE_CSV,
    output_mapping_json_path: Path = FEATURE_CORR_PCA_MAPPING_JSON,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    kernel: str = DEFAULT_KERNEL,
    n_components_per_group: int = DEFAULT_COMPONENTS_PER_GROUP,
    train_end_date: date = TRAIN_END_DATE,
    start_date: date | pd.Timestamp | None = None,
    return_transformed_data: bool = True,
    max_workers: int | None = None,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    DATA_FEATURE_CORR_PCA_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_started_at = time.perf_counter()
    worker_count = resolve_max_workers(max_workers)
    normalized_train_end_date_raw = pd.Timestamp(train_end_date)
    if pd.isna(normalized_train_end_date_raw):
        raise ValueError("train_end_date cannot be NaT.")
    normalized_train_end_date = cast(pd.Timestamp, normalized_train_end_date_raw)

    schema = pq.read_schema(feature_parquet_path)
    feature_columns = build_candidate_feature_columns(schema.names)
    LOGGER.info(
        "Feature corr+pca started: %d candidate features from %s | workers=%d.",
        len(feature_columns),
        feature_parquet_path,
        worker_count,
    )

    correlation_started_at = time.perf_counter()
    correlated_groups = find_correlated_feature_groups_from_parquet(
        feature_parquet_path,
        feature_columns=feature_columns,
        threshold=correlation_threshold,
        train_end_date=normalized_train_end_date.date(),
        start_date=start_date,
        max_workers=worker_count,
    )
    LOGGER.info(
        "Correlation grouping stage completed: %d correlated groups found | elapsed=%.2fs.",
        len(correlated_groups),
        time.perf_counter() - correlation_started_at,
    )

    fit_started_at = time.perf_counter()
    models, applied_groups, skipped_groups = fit_group_models_from_parquet(
        feature_parquet_path=feature_parquet_path,
        correlated_groups=correlated_groups,
        kernel=kernel,
        n_components_per_group=n_components_per_group,
        train_end_date=normalized_train_end_date,
        start_date=start_date,
        max_workers=worker_count,
    )
    LOGGER.info(
        "Kernel PCA model build stage completed: %d models ready | %d skipped groups | elapsed=%.2fs.",
        len(models),
        len(skipped_groups),
        time.perf_counter() - fit_started_at,
    )

    columns_to_drop = {
        member_feature
        for group in applied_groups
        for member_feature in group["member_features"]
    }
    mapping: dict[str, Any] = {
        "correlation_threshold": correlation_threshold,
        "kernel": kernel,
        "n_components_per_group": n_components_per_group,
        "train_end_date": str(normalized_train_end_date.date()),
        "start_date": None if start_date is None else str(pd.Timestamp(start_date).date()),
        "applied_groups": applied_groups,
        "skipped_groups": skipped_groups,
        "retained_original_feature_count": len(feature_columns) - len(columns_to_drop),
    }

    row_count, column_count = stream_transform_to_output(
        feature_parquet_path=feature_parquet_path,
        output_parquet_path=output_parquet_path,
        output_sample_csv_path=output_sample_csv_path,
        columns_to_drop=columns_to_drop,
        models=models,
        transform_feature_fn=transform_group_batch,
        start_date=start_date,
        max_workers=worker_count,
    )
    output_mapping_json_path.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    LOGGER.info(
        "Saved feature corr+pca parquet: %s (%d rows x %d cols)",
        output_parquet_path,
        row_count,
        column_count,
    )
    LOGGER.info("Saved feature corr+pca mapping JSON: %s", output_mapping_json_path)

    transformed: pd.DataFrame | None = None
    if return_transformed_data:
        load_started_at = time.perf_counter()
        transformed = pd.read_parquet(output_parquet_path)
        LOGGER.info(
            "Reloaded transformed corr+pca dataset into memory: %d rows x %d cols | elapsed=%.2fs.",
            len(transformed),
            len(transformed.columns),
            time.perf_counter() - load_started_at,
        )
    LOGGER.info(
        "Feature corr+pca completed in %.2fs.",
        time.perf_counter() - pipeline_started_at,
    )
    return transformed, mapping


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    available_columns = pq.read_schema(FEATURES_OUTPUT_PARQUET).names
    secondary_prediction_start_date = derive_secondary_prediction_start_date(
        FEATURES_OUTPUT_PARQUET,
        available_columns=available_columns,
    )
    run_feature_corr_pca(
        return_transformed_data=False,
        start_date=secondary_prediction_start_date,
    )
    LOGGER.info("Feature correlation + Kernel PCA pipeline completed.")


__all__ = [
    "apply_feature_corr_kernel_pca",
    "build_secondary_prediction_feature_columns",
    "derive_secondary_prediction_start_date",
    "find_correlated_feature_groups",
    "load_feature_dataset",
    "run_feature_corr_pca",
    "save_feature_corr_pca_outputs",
]


if __name__ == "__main__":
    main()
