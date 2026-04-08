from __future__ import annotations

"""Streaming feature pipeline: memory-efficient chunked processing for large universes."""

import gc
import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from core.src.meta_model.features_engineering.config import CROSS_SECTIONAL_BASE_FEATURES, QUANT_FEATURE_PREFIX
from core.src.meta_model.features_engineering.io import (
    iter_parquet_groups,
    load_cleaned_dataset,
    save_lagged_feature_groups,
    write_grouped_parquet,
)
from core.src.meta_model.features_engineering.pipeline import _build_ticker_feature_group
from core.src.meta_model.features_engineering.post_processing import (
    add_calendar_features,
    add_cross_sectional_features,
    add_universe_market_features_for_ticker,
    build_daily_market_aggregates,
    drop_internal_columns,
)
from core.src.meta_model.features_engineering.validation import prepare_input_dataset, validate_base_columns

LOGGER: logging.Logger = logging.getLogger(__name__)


def _stage_path(path: Path, stage_name: str) -> Path:
    return path.with_name(f"{path.stem}.{stage_name}.stage{path.suffix}")


def _cleanup_paths(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _build_base_feature_stage(
    prepared: pd.DataFrame,
    base_stage_path: Path,
) -> pd.DataFrame:
    return_1d_col: str = "_quant_internal_return_1d"
    dollar_volume_col: str = "quant_dollar_volume"
    market_aggregate_frames: list[pd.DataFrame] = []
    total_tickers: int = len(pd.Index(prepared["ticker"]).unique())

    def base_groups():
        for group_index, (_, group) in enumerate(prepared.groupby("ticker", sort=False), start=1):
            enriched_group: pd.DataFrame = _build_ticker_feature_group(group.copy())
            market_aggregate_frames.append(
                pd.DataFrame(
                    enriched_group.loc[:, ["date", return_1d_col, dollar_volume_col]].copy(),
                ),
            )
            if group_index == 1 or group_index % 25 == 0:
                LOGGER.info(
                    "Base feature stage progress: %d/%d tickers processed.",
                    group_index,
                    total_tickers,
                )
            yield enriched_group

    stats = write_grouped_parquet(base_groups(), base_stage_path)
    LOGGER.info(
        "Built base feature stage: %d rows x %d cols across %d tickers.",
        stats["rows"],
        stats["cols"],
        stats["groups"],
    )
    daily_market_aggregates: pd.DataFrame = build_daily_market_aggregates(
        pd.concat(market_aggregate_frames, ignore_index=True),
    )
    del market_aggregate_frames
    gc.collect()
    return daily_market_aggregates


def _build_market_feature_stage(
    base_stage_path: Path,
    market_stage_path: Path,
    daily_market_aggregates: pd.DataFrame,
) -> pd.DataFrame:
    cross_section_frames: list[pd.DataFrame] = []
    total_groups: int = pq.ParquetFile(base_stage_path).num_row_groups

    def market_groups():
        for group_index, base_group in enumerate(iter_parquet_groups(base_stage_path), start=1):
            market_group: pd.DataFrame = add_universe_market_features_for_ticker(
                base_group,
                daily_market_aggregates,
            )
            available_cross_section_columns: list[str] = [
                column
                for column in CROSS_SECTIONAL_BASE_FEATURES
                if column in market_group.columns
            ]
            cross_section_frames.append(
                pd.DataFrame(
                    market_group.loc[:, ["date", "ticker", *available_cross_section_columns]].copy(),
                ),
            )
            if group_index == 1 or group_index % 25 == 0 or group_index == total_groups:
                LOGGER.info(
                    "Market feature stage progress: %d/%d tickers processed.",
                    group_index,
                    total_groups,
                )
            yield market_group

    stats = write_grouped_parquet(market_groups(), market_stage_path)
    LOGGER.info(
        "Built market feature stage: %d rows x %d cols across %d tickers.",
        stats["rows"],
        stats["cols"],
        stats["groups"],
    )
    cross_section_source: pd.DataFrame = pd.concat(cross_section_frames, ignore_index=True)
    del cross_section_frames
    gc.collect()
    return cross_section_source


def _build_cross_sectional_stage(
    cross_section_source: pd.DataFrame,
    cross_section_stage_path: Path,
) -> None:
    cross_section_source["ticker"] = cross_section_source["ticker"].astype("category")
    cross_section_enriched: pd.DataFrame = add_cross_sectional_features(
        cross_section_source.sort_values(["date", "ticker"]).reset_index(drop=True),
    )
    cross_section_columns: list[str] = [
        "date",
        "ticker",
        *[
            column
            for column in cross_section_enriched.columns
            if column.startswith(f"{QUANT_FEATURE_PREFIX}cs_")
        ],
    ]
    thin_cross_section = pd.DataFrame(cross_section_enriched.loc[:, cross_section_columns]).sort_values(
        by=["ticker", "date"],
    ).reset_index(drop=True)
    write_grouped_parquet(
        (group.copy() for _, group in thin_cross_section.groupby("ticker", sort=False)),
        cross_section_stage_path,
    )
    LOGGER.info(
        "Built cross-sectional stage: %d rows x %d cols.",
        len(thin_cross_section),
        len(thin_cross_section.columns),
    )


def _iter_final_feature_groups(
    market_stage_path: Path,
    cross_section_stage_path: Path,
):
    market_groups = iter_parquet_groups(market_stage_path)
    cross_section_group_map: dict[str, pd.DataFrame] = {}
    for cross_section_group in iter_parquet_groups(cross_section_stage_path):
        ticker = str(cross_section_group["ticker"].iloc[0])
        cross_section_group_map[ticker] = cross_section_group.reset_index(drop=True)

    for market_group in market_groups:
        ticker = str(market_group["ticker"].iloc[0])
        cross_section_group = cross_section_group_map.get(ticker)
        if cross_section_group is None:
            raise RuntimeError(f"Missing cross-sectional stage for ticker {ticker}.")

        market_dates = pd.to_datetime(market_group["date"]).reset_index(drop=True)
        cross_section_dates = pd.to_datetime(cross_section_group["date"]).reset_index(drop=True)
        market_tickers = market_group["ticker"].astype(str).reset_index(drop=True)
        cross_section_tickers = cross_section_group["ticker"].astype(str).reset_index(drop=True)
        if not market_dates.equals(cross_section_dates) or not market_tickers.equals(cross_section_tickers):
            raise RuntimeError(
                f"Market and cross-sectional stages are misaligned by (date, ticker) for ticker {ticker}.",
            )

        final_group: pd.DataFrame = market_group.reset_index(drop=True).copy()
        for column in cross_section_group.columns:
            if column in {"date", "ticker"}:
                continue
            final_group[column] = cross_section_group[column].to_numpy()
        final_group = add_calendar_features(final_group)
        final_group = drop_internal_columns(final_group)
        yield final_group.sort_values(["ticker", "date"]).reset_index(drop=True)


def run_feature_engineering_pipeline(
    cleaned_path: Path,
    parquet_path: Path,
    csv_path: Path,
) -> dict[str, Path]:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    base_stage_path: Path = _stage_path(parquet_path, "base")
    market_stage_path: Path = _stage_path(parquet_path, "market")
    cross_section_stage_path: Path = _stage_path(parquet_path, "cross_section")
    stage_paths: list[Path] = [base_stage_path, market_stage_path, cross_section_stage_path]
    _cleanup_paths(stage_paths)

    cleaned: pd.DataFrame = load_cleaned_dataset(cleaned_path)
    validate_base_columns(cleaned)
    prepared: pd.DataFrame = prepare_input_dataset(cleaned)
    expected_rows: int = len(prepared)
    expected_tickers: int = len(pd.Index(prepared["ticker"]).unique())
    LOGGER.info(
        "Starting streaming feature engineering pipeline for %d rows across %d tickers.",
        expected_rows,
        expected_tickers,
    )

    try:
        daily_market_aggregates = _build_base_feature_stage(prepared, base_stage_path)
        del cleaned
        del prepared
        gc.collect()

        cross_section_source = _build_market_feature_stage(
            base_stage_path,
            market_stage_path,
            daily_market_aggregates,
        )
        del daily_market_aggregates
        gc.collect()

        _build_cross_sectional_stage(cross_section_source, cross_section_stage_path)
        del cross_section_source
        gc.collect()

        result = save_lagged_feature_groups(
            _iter_final_feature_groups(market_stage_path, cross_section_stage_path),
            parquet_path,
            csv_path,
            expected_rows=expected_rows,
            expected_tickers=expected_tickers,
        )
    finally:
        _cleanup_paths(stage_paths)

    LOGGER.info("Streaming feature engineering pipeline completed.")
    return result
