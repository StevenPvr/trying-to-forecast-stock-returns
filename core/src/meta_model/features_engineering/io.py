from __future__ import annotations

"""Feature-engineering I/O: chunked Parquet reads, lagged feature writes, and dataset assembly."""

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from core.src.meta_model.data.constants import RANDOM_SEED, SAMPLE_FRAC
from core.src.meta_model.features_engineering.lag_features import (
    add_feature_lags,
    build_lagged_feature_group,
    get_laggable_feature_columns,
)

LOGGER: logging.Logger = logging.getLogger(__name__)


def _to_parquet_table_without_pandas_metadata(data: pd.DataFrame) -> pa.Table:
    table: pa.Table = pa.Table.from_pandas(data, preserve_index=False)
    return table.replace_schema_metadata()


def _inprogress_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.inprogress{path.suffix}")


def _ensure_parent_dirs(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def _reset_inprogress_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def write_grouped_parquet(groups: Iterable[pd.DataFrame], parquet_path: Path) -> dict[str, int]:
    _ensure_parent_dirs(parquet_path)
    writer: pq.ParquetWriter | None = None
    total_rows: int = 0
    total_groups: int = 0
    total_cols: int = 0
    try:
        for group in groups:
            table: pa.Table = _to_parquet_table_without_pandas_metadata(group)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, table.schema, compression="snappy")
                total_cols = len(group.columns)
            writer.write_table(table)
            total_rows += len(group)
            total_groups += 1
    finally:
        if writer is not None:
            writer.close()

    return {
        "rows": total_rows,
        "groups": total_groups,
        "cols": total_cols,
    }


def iter_parquet_groups(parquet_path: Path) -> Iterable[pd.DataFrame]:
    parquet_file = pq.ParquetFile(parquet_path)
    for row_group_index in range(parquet_file.num_row_groups):
        yield parquet_file.read_row_group(row_group_index).to_pandas()


def load_cleaned_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input parquet not found: {path}")
    df: pd.DataFrame = pd.read_parquet(path)
    LOGGER.info("Loaded cleaned dataset: %d rows x %d cols", len(df), len(df.columns))
    return df


def save_feature_dataset(
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
        "Saved feature parquet: %s (%d rows x %d cols)",
        parquet_path,
        len(data),
        len(data.columns),
    )
    LOGGER.info("Saved feature sample CSV: %s", csv_path)
    return {"parquet": parquet_path, "sample_csv": csv_path}


def save_lagged_feature_dataset(
    data: pd.DataFrame,
    parquet_path: Path,
    csv_path: Path,
) -> dict[str, Path]:
    expected_rows: int = len(data)
    expected_tickers: int = len(pd.Index(data["ticker"]).unique()) if "ticker" in data.columns else 0
    if data.empty:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        empty_lagged: pd.DataFrame = add_feature_lags(data)
        empty_lagged.to_parquet(parquet_path, index=False)
        empty_lagged.to_csv(csv_path, index=False)
        return {"parquet": parquet_path, "sample_csv": csv_path}

    sorted_data: pd.DataFrame = data.sort_values(["ticker", "date"]).reset_index(drop=True)
    groups = (group for _, group in sorted_data.groupby("ticker", sort=False))
    return save_lagged_feature_groups(
        groups,
        parquet_path,
        csv_path,
        expected_rows=expected_rows,
        expected_tickers=expected_tickers,
    )


def save_lagged_feature_groups(
    groups: Iterable[pd.DataFrame],
    parquet_path: Path,
    csv_path: Path,
    *,
    expected_rows: int,
    expected_tickers: int,
) -> dict[str, Path]:
    _ensure_parent_dirs(parquet_path, csv_path)
    parquet_tmp_path: Path = _inprogress_path(parquet_path)
    csv_tmp_path: Path = _inprogress_path(csv_path)
    _reset_inprogress_outputs(parquet_tmp_path, csv_tmp_path)

    writer: pq.ParquetWriter | None = None
    sample_header_written: bool = False
    empty_sample_template: pd.DataFrame | None = None
    total_rows: int = 0
    total_cols: int = 0
    total_tickers_written: int = 0
    laggable_columns: list[str] | None = None

    LOGGER.info(
        "Saving lagged feature artifacts for %d rows across %d tickers to in-progress files.",
        expected_rows,
        expected_tickers,
    )

    try:
        for group_index, group in enumerate(groups):
            if laggable_columns is None:
                laggable_columns = get_laggable_feature_columns(list(group.columns), group)
            lagged_group: pd.DataFrame = build_lagged_feature_group(
                group,
                laggable_columns=laggable_columns,
            )
            if total_cols == 0:
                total_cols = len(lagged_group.columns)
                empty_sample_template = lagged_group.iloc[0:0].copy()
            table: pa.Table = _to_parquet_table_without_pandas_metadata(lagged_group)
            if writer is None:
                writer = pq.ParquetWriter(parquet_tmp_path, table.schema, compression="snappy")
            writer.write_table(table)
            total_rows += len(lagged_group)
            total_tickers_written += 1

            sample_chunk: pd.DataFrame = lagged_group.sample(
                frac=SAMPLE_FRAC,
                random_state=RANDOM_SEED + group_index,
            )
            if sample_chunk.empty:
                continue
            sample_chunk = sample_chunk.sort_values(["date", "ticker"]).reset_index(drop=True)
            write_mode: str = "a" if sample_header_written else "w"
            sample_chunk.to_csv(
                csv_tmp_path,
                index=False,
                mode=write_mode,
                header=not sample_header_written,
            )
            sample_header_written = True
            if (
                total_tickers_written == 1
                or total_tickers_written % 25 == 0
                or total_tickers_written == expected_tickers
            ):
                LOGGER.info(
                    "Lagged artifact save progress: %d/%d tickers written, %d/%d rows materialized.",
                    total_tickers_written,
                    expected_tickers,
                    total_rows,
                    expected_rows,
                )
    except Exception:
        if writer is not None:
            writer.close()
            writer = None
        LOGGER.exception(
            "Lagged artifact save failed after %d/%d tickers. In-progress files left in place for inspection: %s and %s",
            total_tickers_written,
            expected_tickers,
            parquet_tmp_path,
            csv_tmp_path,
        )
        raise
    finally:
        if writer is not None:
            writer.close()

    if not sample_header_written:
        if empty_sample_template is None:
            raise RuntimeError("No lagged groups were produced during artifact save.")
        empty_sample_template.to_csv(csv_tmp_path, index=False)

    if total_rows != expected_rows or total_tickers_written != expected_tickers:
        raise RuntimeError(
            "Lagged artifact save did not cover the full dataset before publish: "
            f"rows {total_rows}/{expected_rows}, tickers {total_tickers_written}/{expected_tickers}.",
        )

    parquet_tmp_path.replace(parquet_path)
    csv_tmp_path.replace(csv_path)

    LOGGER.info(
        "Saved lagged feature parquet: %s (%d rows x %d cols)",
        parquet_path,
        total_rows,
        total_cols,
    )
    LOGGER.info("Saved lagged feature sample CSV: %s", csv_path)
    return {"parquet": parquet_path, "sample_csv": csv_path}
