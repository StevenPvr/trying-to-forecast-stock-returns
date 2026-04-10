from __future__ import annotations

"""Feature pipeline: parallel per-ticker feature generation and assembly."""

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from core.src.meta_model.broker_xtb.specs import BrokerSpecProvider
from core.src.meta_model.data.paths import REFERENCE_EARNINGS_HISTORY_CSV
from core.src.meta_model.features_engineering.config import (
    DEEP_FEATURE_PREFIX,
    QUANT_FEATURE_PREFIX,
    TA_FEATURE_PREFIX,
)
from core.src.meta_model.runtime_parallelism import resolve_requested_worker_count
from core.src.meta_model.features_engineering.deep.price_features import add_deep_price_features_for_ticker
from core.src.meta_model.features_engineering.high_level_features import add_high_level_features
from core.src.meta_model.features_engineering.post_processing import (
    add_calendar_features,
    add_cross_sectional_features,
    add_universe_market_features,
    drop_internal_columns,
)
from core.src.meta_model.features_engineering.quant_features import add_quant_features_for_ticker
from core.src.meta_model.features_engineering.ta_features import add_ta_features_for_ticker
from core.src.meta_model.features_engineering.validation import (
    prepare_input_dataset,
    validate_base_columns,
    validate_output_dataset,
)

LOGGER: logging.Logger = logging.getLogger(__name__)


def _build_ticker_feature_group(group: pd.DataFrame) -> pd.DataFrame:
    ta_enriched: pd.DataFrame = add_ta_features_for_ticker(group)
    quant_enriched: pd.DataFrame = add_quant_features_for_ticker(ta_enriched)
    return add_deep_price_features_for_ticker(quant_enriched)


def resolve_max_workers(requested_workers: int | None = None) -> int:
    return resolve_requested_worker_count(requested_workers)


def iter_ticker_feature_groups(
    prepared: pd.DataFrame,
    max_workers: int | None = None,
):
    ticker_groups = prepared.groupby("ticker", sort=False)
    ticker_count = len(pd.Index(prepared["ticker"]).unique())
    worker_count = min(resolve_max_workers(max_workers), max(1, ticker_count))
    groups = [group.copy() for _, group in ticker_groups]

    if worker_count == 1:
        LOGGER.info(
            "Sequential ticker feature engineering across %d tickers",
            ticker_count,
        )
        for group in groups:
            yield _build_ticker_feature_group(group)
        return

    LOGGER.info(
        "Parallel ticker feature engineering across %d tickers with %d workers",
        ticker_count,
        worker_count,
    )
    try:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for group in executor.map(_build_ticker_feature_group, groups):
                yield group
    except (NotImplementedError, PermissionError, OSError) as exc:
        LOGGER.warning(
            "Falling back to sequential feature engineering because process workers are unavailable: %s",
            exc,
        )
        for group in groups:
            yield _build_ticker_feature_group(group)


def build_feature_dataset(
    df: pd.DataFrame,
    max_workers: int | None = None,
    *,
    earnings_path: Path = REFERENCE_EARNINGS_HISTORY_CSV,
    spec_provider: BrokerSpecProvider | None = None,
) -> pd.DataFrame:
    validate_base_columns(df)
    prepared: pd.DataFrame = prepare_input_dataset(df)

    enriched_groups: list[pd.DataFrame] = list(
        iter_ticker_feature_groups(prepared, max_workers=max_workers),
    )
    enriched: pd.DataFrame = pd.concat(enriched_groups, ignore_index=True)
    enriched = enriched.sort_values(["date", "ticker"]).reset_index(drop=True)
    enriched = add_universe_market_features(enriched)
    enriched = add_cross_sectional_features(enriched)
    enriched = add_calendar_features(enriched)
    enriched = add_high_level_features(
        enriched,
        earnings_path=earnings_path,
        spec_provider=spec_provider,
    )
    enriched = drop_internal_columns(enriched)
    enriched = enriched.sort_values(["date", "ticker"]).reset_index(drop=True)

    validate_output_dataset(enriched)

    ta_columns_added: int = len(
        [column for column in enriched.columns if column.startswith(TA_FEATURE_PREFIX)],
    )
    quant_columns_added: int = len(
        [column for column in enriched.columns if column.startswith(QUANT_FEATURE_PREFIX)],
    )
    deep_columns_added: int = len(
        [column for column in enriched.columns if column.startswith(DEEP_FEATURE_PREFIX)],
    )
    LOGGER.info(
        "Built feature dataset: %d rows x %d cols (%d TA columns, %d quant columns, %d deep columns)",
        len(enriched),
        len(enriched.columns),
        ta_columns_added,
        quant_columns_added,
        deep_columns_added,
    )
    return enriched


def build_ta_feature_dataset(
    df: pd.DataFrame,
    max_workers: int | None = None,
) -> pd.DataFrame:
    return build_feature_dataset(df, max_workers=max_workers)
