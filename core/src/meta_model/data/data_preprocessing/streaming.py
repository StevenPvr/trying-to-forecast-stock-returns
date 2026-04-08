from __future__ import annotations

"""Streaming preprocessor: chunk-based target and feature computation for large datasets."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

from core.src.meta_model.data.constants import RANDOM_SEED, SAMPLE_FRAC
from core.src.meta_model.data.data_preprocessing.main import (
    SPLIT_COLUMN,
    TARGET_RELATED_COLUMNS,
    apply_target_metric_panel,
    assign_dataset_splits,
    build_protected_columns,
    build_feature_fill_limits,
    build_target_metric_panel,
    create_target_main_group,
    filter_from_start_date,
    forward_fill_features_by_ticker,
    remove_rows_with_missing_values,
    save_preprocessing_contract_artifacts_from_columns,
    validate_required_columns_not_missing,
)
from core.src.meta_model.data.paths import (
    PREPROCESSED_OUTPUT_PARQUET,
    PREPROCESSED_OUTPUT_SAMPLE_CSV,
    PREPROCESSED_RESEARCH_LABEL_PANEL_PARQUET,
    PREPROCESSED_TEST_PARQUET,
    PREPROCESSED_TEST_SAMPLE_CSV,
    PREPROCESSED_TRAIN_PARQUET,
    PREPROCESSED_TRAIN_SAMPLE_CSV,
    PREPROCESSED_VAL_PARQUET,
    PREPROCESSED_VAL_SAMPLE_CSV,
)
from core.src.meta_model.data.registry import build_feature_registry_from_columns
from core.src.meta_model.features_engineering.io import _to_parquet_table_without_pandas_metadata
from core.src.meta_model.model_contract import (
    INTRADAY_NET_RETURN_COLUMN,
    MODEL_TARGET_COLUMN,
    REALIZED_RETURN_COLUMN,
    WEEK_HOLD_NET_RETURN_COLUMN,
)

LOGGER: logging.Logger = logging.getLogger(__name__)


def _parquet_columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema_arrow.names)


def _inprogress_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.inprogress{path.suffix}")


def _ensure_parent_dirs(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def _unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _delete_if_exists(*paths: Path) -> None:
    for path in paths:
        _unlink_if_exists(path)


def _available_columns(path: Path, requested_columns: list[str]) -> list[str]:
    available = set(_parquet_columns(path))
    return [column for column in requested_columns if column in available]


def _normalize_group_order(group: pd.DataFrame) -> pd.DataFrame:
    if "date" in group.columns:
        ordered = group.sort_values(["ticker", "date"]).reset_index(drop=True)
        ordered["date"] = pd.to_datetime(ordered["date"])
        return pd.DataFrame(ordered)
    return pd.DataFrame(group.sort_values(["ticker"]).reset_index(drop=True))


def _finalize_pending_group(ticker: str, pending_frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(pending_frames, ignore_index=True)
    if "date" in combined.columns:
        combined = combined.sort_values(["date"]).reset_index(drop=True)
        combined["date"] = pd.to_datetime(combined["date"])
    if combined["ticker"].astype(str).nunique() != 1:
        raise ValueError(f"Ticker group assembly failed for {ticker}: multiple tickers found.")
    return pd.DataFrame(combined)


def iter_ticker_groups_from_parquet(
    parquet_path: Path,
    *,
    columns: list[str] | None = None,
) -> Iterable[pd.DataFrame]:
    parquet_file = pq.ParquetFile(parquet_path)
    pending_ticker: str | None = None
    pending_frames: list[pd.DataFrame] = []
    closed_tickers: set[str] = set()

    for row_group_index in range(parquet_file.num_row_groups):
        group = parquet_file.read_row_group(row_group_index, columns=columns).to_pandas()
        if group.empty:
            continue
        if "ticker" not in group.columns:
            raise ValueError(f"Ticker column is required to iterate parquet groups: {parquet_path}")
        normalized = _normalize_group_order(group)
        for ticker_value, ticker_group in normalized.groupby("ticker", sort=False):
            ticker = str(ticker_value)
            ticker_frame = pd.DataFrame(ticker_group.reset_index(drop=True))
            if pending_ticker is None:
                pending_ticker = ticker
                pending_frames = [ticker_frame]
                continue
            if ticker == pending_ticker:
                pending_frames.append(ticker_frame)
                continue
            if ticker in closed_tickers:
                raise ValueError(
                    "Input parquet is not grouped contiguously by ticker across row groups. "
                    f"Ticker {ticker} reappeared after being finalized.",
                )
            if pending_ticker is None:
                raise RuntimeError("Pending ticker unexpectedly missing during parquet iteration.")
            yield _finalize_pending_group(pending_ticker, pending_frames)
            closed_tickers.add(pending_ticker)
            pending_ticker = ticker
            pending_frames = [ticker_frame]

    if pending_ticker is not None:
        yield _finalize_pending_group(pending_ticker, pending_frames)


@dataclass(frozen=True)
class StageWriteSummary:
    rows: int
    tickers: int
    cols: int


@dataclass
class PreprocessingColumnStats:
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    missing_count_by_column: dict[str, int] = field(default_factory=dict)
    non_null_count_by_column: dict[str, int] = field(default_factory=dict)

    def ensure_columns(self, columns: list[str]) -> None:
        if self.columns:
            return
        self.columns = list(columns)
        self.missing_count_by_column = {column: 0 for column in self.columns}
        self.non_null_count_by_column = {column: 0 for column in self.columns}

    def update(self, data: pd.DataFrame) -> None:
        self.ensure_columns(list(data.columns))
        missing_by_column = data.isna().sum()
        non_null_by_column = data.notna().sum()
        self.row_count += len(data)
        for column_name in self.columns:
            self.missing_count_by_column[column_name] += int(missing_by_column.get(column_name, 0))
            self.non_null_count_by_column[column_name] += int(non_null_by_column.get(column_name, 0))


class IncrementalDatasetWriter:
    def __init__(
        self,
        parquet_path: Path,
        csv_path: Path | None = None,
        *,
        log_name: str,
        sample_seed_offset: int = 0,
    ) -> None:
        self.parquet_path = parquet_path
        self.csv_path = csv_path
        self.log_name = log_name
        self.sample_seed_offset = sample_seed_offset
        self.parquet_tmp_path = _inprogress_path(parquet_path)
        self.csv_tmp_path = _inprogress_path(csv_path) if csv_path is not None else None
        self.writer: pq.ParquetWriter | None = None
        self.sample_header_written = False
        self.empty_template: pd.DataFrame | None = None
        self.total_rows = 0
        self.total_groups = 0
        self.total_cols = 0

    def reset(self) -> None:
        paths = [self.parquet_tmp_path]
        if self.csv_tmp_path is not None:
            paths.append(self.csv_tmp_path)
        _ensure_parent_dirs(*paths)
        _delete_if_exists(*paths)

    def write(self, data: pd.DataFrame, group_index: int) -> None:
        if data.empty:
            if self.empty_template is None:
                self.empty_template = data.iloc[0:0].copy()
            return
        table = _to_parquet_table_without_pandas_metadata(data)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.parquet_tmp_path, table.schema, compression="snappy")
            self.total_cols = len(data.columns)
            self.empty_template = data.iloc[0:0].copy()
        self.writer.write_table(table)
        self.total_rows += len(data)
        self.total_groups += 1
        self._write_sample(data, group_index)

    def _write_sample(self, data: pd.DataFrame, group_index: int) -> None:
        if self.csv_tmp_path is None:
            return
        sample = data.sample(
            frac=SAMPLE_FRAC,
            random_state=RANDOM_SEED + self.sample_seed_offset + group_index,
        )
        if sample.empty:
            return
        sample = sample.sort_values(["date", "ticker"]).reset_index(drop=True)
        sample.to_csv(
            self.csv_tmp_path,
            index=False,
            mode="a" if self.sample_header_written else "w",
            header=not self.sample_header_written,
        )
        self.sample_header_written = True

    def publish(self, *, empty_columns: list[str] | None = None) -> StageWriteSummary:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.total_rows == 0:
            template = self.empty_template
            if template is None:
                if empty_columns is None:
                    raise RuntimeError(f"{self.log_name} produced no rows and no empty schema was provided.")
                template = pd.DataFrame(columns=empty_columns)
            template.to_parquet(self.parquet_tmp_path, index=False)
            if self.csv_tmp_path is not None:
                template.to_csv(self.csv_tmp_path, index=False)
        elif self.csv_tmp_path is not None and not self.sample_header_written:
            if self.empty_template is None:
                raise RuntimeError(f"{self.log_name} has data but no sample template.")
            self.empty_template.to_csv(self.csv_tmp_path, index=False)

        self.parquet_tmp_path.replace(self.parquet_path)
        if self.csv_tmp_path is not None:
            self.csv_tmp_path.replace(self.csv_path)
        LOGGER.info(
            "Published %s: %s (%d rows x %d cols)",
            self.log_name,
            self.parquet_path,
            self.total_rows,
            self.total_cols if self.total_cols > 0 else len(empty_columns or []),
        )
        return StageWriteSummary(
            rows=self.total_rows,
            tickers=self.total_groups,
            cols=self.total_cols if self.total_cols > 0 else len(empty_columns or []),
        )


def _build_stage1_group(
    group: pd.DataFrame,
    *,
    feature_fill_limits: dict[str, int | None],
    protected_columns: list[str],
) -> pd.DataFrame:
    filtered = filter_from_start_date(group)
    if filtered.empty:
        return filtered
    targeted = create_target_main_group(filtered)
    split_ready = assign_dataset_splits(targeted)
    if split_ready.empty:
        return split_ready
    return forward_fill_features_by_ticker(
        split_ready,
        feature_fill_limits=feature_fill_limits,
        protected_columns=protected_columns,
    )


def write_stage1_dataset(
    input_path: Path,
    stage1_path: Path,
) -> StageWriteSummary:
    source_columns = _parquet_columns(input_path)
    feature_registry = build_feature_registry_from_columns(source_columns)
    feature_fill_limits = build_feature_fill_limits(feature_registry)
    protected_columns = build_protected_columns()
    writer = IncrementalDatasetWriter(stage1_path, None, log_name="preprocessing stage1")
    writer.reset()
    ticker_count = 0
    for ticker_count, group in enumerate(iter_ticker_groups_from_parquet(input_path), start=1):
        stage1_group = _build_stage1_group(
            group,
            feature_fill_limits=feature_fill_limits,
            protected_columns=protected_columns,
        )
        writer.write(stage1_group, ticker_count)
        if ticker_count == 1 or ticker_count % 25 == 0:
            LOGGER.info(
                "Stage1 progress: %d tickers processed, %d rows written so far.",
                ticker_count,
                writer.total_rows,
            )
    summary = writer.publish(empty_columns=source_columns)
    if summary.rows <= 0:
        raise RuntimeError("Stage1 preprocessing produced no rows.")
    return summary


def compute_target_metric_table(
    stage1_path: Path,
    metrics_path: Path,
) -> pd.DataFrame:
    metric_columns = _available_columns(
        stage1_path,
        [
            "date",
            "ticker",
            "company_sector",
            "company_industry",
            INTRADAY_NET_RETURN_COLUMN,
            WEEK_HOLD_NET_RETURN_COLUMN,
        ],
    )
    stage1_metrics = pd.read_parquet(stage1_path, columns=metric_columns)
    stage1_metrics["date"] = pd.to_datetime(stage1_metrics["date"])
    metric_panel = build_target_metric_panel(stage1_metrics)
    metrics_tmp_path = _inprogress_path(metrics_path)
    _ensure_parent_dirs(metrics_tmp_path)
    _unlink_if_exists(metrics_tmp_path)
    metric_panel.to_parquet(metrics_tmp_path, index=False)
    metrics_tmp_path.replace(metrics_path)
    LOGGER.info(
        "Built target metric panel: %s (%d rows x %d cols)",
        metrics_path,
        len(metric_panel),
        len(metric_panel.columns),
    )
    return metric_panel


def write_stage2_dataset(
    stage1_path: Path,
    metric_panel: pd.DataFrame,
    stage2_path: Path,
) -> PreprocessingColumnStats:
    writer = IncrementalDatasetWriter(stage2_path, None, log_name="preprocessing stage2")
    writer.reset()
    stats = PreprocessingColumnStats()
    metric_groups = iter(metric_panel.groupby("ticker", sort=False))
    for group_index, stage1_group in enumerate(iter_ticker_groups_from_parquet(stage1_path), start=1):
        ticker = str(stage1_group["ticker"].iloc[0])
        metric_entry = next(metric_groups, None)
        if metric_entry is None:
            raise RuntimeError(f"Missing target metrics for ticker {ticker}.")
        metric_ticker, ticker_metrics = metric_entry
        if str(metric_ticker) != ticker:
            raise RuntimeError(
                f"Ticker ordering mismatch between stage1 and metric panel: {ticker} != {metric_ticker}",
            )
        merged = apply_target_metric_panel(
            stage1_group,
            pd.DataFrame(ticker_metrics.reset_index(drop=True)),
        )
        cleaned = remove_rows_with_missing_values(
            merged,
            required_columns=TARGET_RELATED_COLUMNS,
        )
        if cleaned.empty:
            continue
        stats.update(cleaned)
        writer.write(cleaned, group_index)
        if group_index == 1 or group_index % 25 == 0:
            LOGGER.info(
                "Stage2 progress: %d tickers processed, %d rows retained so far.",
                group_index,
                writer.total_rows,
            )

    leftover_metrics = next(metric_groups, None)
    if leftover_metrics is not None:
        raise RuntimeError(
            f"Metric panel still has unconsumed ticker groups after stage2: {leftover_metrics[0]}",
        )
    writer.publish(empty_columns=stats.columns)
    if stats.row_count <= 0:
        raise RuntimeError("Stage2 preprocessing produced no rows.")
    return stats


def resolve_final_columns(
    stats: PreprocessingColumnStats,
    *,
    protected_columns: list[str],
) -> list[str]:
    protected = set(protected_columns)
    final_columns: list[str] = []
    for column_name in stats.columns:
        if column_name in protected:
            final_columns.append(column_name)
            continue
        if stats.non_null_count_by_column.get(column_name, 0) <= 0:
            continue
        final_columns.append(column_name)
    return final_columns


def _validate_required_columns_from_stats(
    stats: PreprocessingColumnStats,
    *,
    required_columns: list[str],
    final_columns: list[str],
) -> None:
    final_column_set = set(final_columns)
    missing_columns = [column for column in required_columns if column not in final_column_set]
    if missing_columns:
        raise ValueError(
            "Required columns are missing from final preprocessed outputs: "
            + ", ".join(sorted(missing_columns)),
        )
    invalid_columns = [
        column
        for column in required_columns
        if stats.missing_count_by_column.get(column, 0) > 0
        or stats.non_null_count_by_column.get(column, 0) <= 0
    ]
    if invalid_columns:
        details = ", ".join(
            f"{column}=missing:{stats.missing_count_by_column.get(column, 0)}"
            for column in invalid_columns
        )
        raise ValueError(f"Required columns still contain missing values after stage2: {details}")


def publish_final_outputs(
    stage2_path: Path,
    *,
    final_columns: list[str],
    main_parquet_path: Path = PREPROCESSED_OUTPUT_PARQUET,
    main_csv_path: Path = PREPROCESSED_OUTPUT_SAMPLE_CSV,
    train_parquet_path: Path = PREPROCESSED_TRAIN_PARQUET,
    train_csv_path: Path = PREPROCESSED_TRAIN_SAMPLE_CSV,
    val_parquet_path: Path = PREPROCESSED_VAL_PARQUET,
    val_csv_path: Path = PREPROCESSED_VAL_SAMPLE_CSV,
    test_parquet_path: Path = PREPROCESSED_TEST_PARQUET,
    test_csv_path: Path = PREPROCESSED_TEST_SAMPLE_CSV,
    label_panel_path: Path = PREPROCESSED_RESEARCH_LABEL_PANEL_PARQUET,
) -> None:
    main_writer = IncrementalDatasetWriter(
        main_parquet_path,
        main_csv_path,
        log_name="preprocessed dataset",
        sample_seed_offset=0,
    )
    label_columns = [column for column in ["date", "ticker", *TARGET_RELATED_COLUMNS] if column in final_columns]
    label_writer = IncrementalDatasetWriter(
        label_panel_path,
        None,
        log_name="research label panel",
    )
    split_writers: dict[str, IncrementalDatasetWriter] = {
        "train": IncrementalDatasetWriter(
            train_parquet_path,
            train_csv_path,
            log_name="preprocessed train split",
            sample_seed_offset=1_000,
        ),
        "val": IncrementalDatasetWriter(
            val_parquet_path,
            val_csv_path,
            log_name="preprocessed val split",
            sample_seed_offset=2_000,
        ),
        "test": IncrementalDatasetWriter(
            test_parquet_path,
            test_csv_path,
            log_name="preprocessed test split",
            sample_seed_offset=3_000,
        ),
    }

    main_writer.reset()
    label_writer.reset()
    for writer in split_writers.values():
        writer.reset()

    for group_index, stage2_group in enumerate(
        iter_ticker_groups_from_parquet(stage2_path, columns=final_columns),
        start=1,
    ):
        final_group = pd.DataFrame(stage2_group.loc[:, final_columns].copy())
        main_writer.write(final_group, group_index)
        label_writer.write(
            pd.DataFrame(final_group.loc[:, label_columns].copy()),
            group_index,
        )
        for split_name, split_writer in split_writers.items():
            split_group = pd.DataFrame(final_group.loc[final_group[SPLIT_COLUMN] == split_name].copy())
            split_writer.write(split_group, group_index)

    main_writer.publish(empty_columns=final_columns)
    label_writer.publish(empty_columns=label_columns)
    for split_name, split_writer in split_writers.items():
        split_writer.publish(empty_columns=final_columns)
        LOGGER.info("Published split output for %s.", split_name)


def run_streaming_preprocessing(
    input_path: Path,
    *,
    output_parquet_path: Path = PREPROCESSED_OUTPUT_PARQUET,
    output_csv_path: Path = PREPROCESSED_OUTPUT_SAMPLE_CSV,
    train_parquet_path: Path = PREPROCESSED_TRAIN_PARQUET,
    train_csv_path: Path = PREPROCESSED_TRAIN_SAMPLE_CSV,
    val_parquet_path: Path = PREPROCESSED_VAL_PARQUET,
    val_csv_path: Path = PREPROCESSED_VAL_SAMPLE_CSV,
    test_parquet_path: Path = PREPROCESSED_TEST_PARQUET,
    test_csv_path: Path = PREPROCESSED_TEST_SAMPLE_CSV,
    label_panel_path: Path = PREPROCESSED_RESEARCH_LABEL_PANEL_PARQUET,
) -> None:
    stage1_path = output_parquet_path.with_name(
        f"{output_parquet_path.stem}.stage1.parquet",
    )
    stage2_path = output_parquet_path.with_name(
        f"{output_parquet_path.stem}.stage2.parquet",
    )
    metrics_path = output_parquet_path.with_name(
        f"{output_parquet_path.stem}.target_metrics.parquet",
    )

    _delete_if_exists(stage1_path, stage2_path, metrics_path)

    write_stage1_dataset(input_path, stage1_path)
    metric_panel = compute_target_metric_table(stage1_path, metrics_path)
    stats = write_stage2_dataset(stage1_path, metric_panel, stage2_path)
    _delete_if_exists(stage1_path, metrics_path)

    final_columns = resolve_final_columns(
        stats,
        protected_columns=build_protected_columns(),
    )
    _validate_required_columns_from_stats(
        stats,
        required_columns=["date", "ticker", SPLIT_COLUMN, MODEL_TARGET_COLUMN, REALIZED_RETURN_COLUMN],
        final_columns=final_columns,
    )
    publish_final_outputs(
        stage2_path,
        final_columns=final_columns,
        main_parquet_path=output_parquet_path,
        main_csv_path=output_csv_path,
        train_parquet_path=train_parquet_path,
        train_csv_path=train_csv_path,
        val_parquet_path=val_parquet_path,
        val_csv_path=val_csv_path,
        test_parquet_path=test_parquet_path,
        test_csv_path=test_csv_path,
        label_panel_path=label_panel_path,
    )
    save_preprocessing_contract_artifacts_from_columns(final_columns)

    published = pd.read_parquet(
        output_parquet_path,
        columns=["date", "ticker", SPLIT_COLUMN, MODEL_TARGET_COLUMN, REALIZED_RETURN_COLUMN],
    )
    validate_required_columns_not_missing(
        published,
        required_columns=["date", "ticker", SPLIT_COLUMN, MODEL_TARGET_COLUMN, REALIZED_RETURN_COLUMN],
    )
    _delete_if_exists(stage2_path)


__all__ = [
    "IncrementalDatasetWriter",
    "PreprocessingColumnStats",
    "StageWriteSummary",
    "compute_target_metric_table",
    "iter_ticker_groups_from_parquet",
    "publish_final_outputs",
    "resolve_final_columns",
    "run_streaming_preprocessing",
    "write_stage1_dataset",
    "write_stage2_dataset",
]
