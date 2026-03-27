from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from core.src.meta_model.data.paths import (
    DATA_EVALUATE_DIR,
    EVALUATE_BACKTEST_DAILY_CSV,
    EVALUATE_BACKTEST_DAILY_PARQUET,
    EVALUATE_BACKTEST_SUMMARY_JSON,
    EVALUATE_BACKTEST_TRADES_CSV,
    EVALUATE_BACKTEST_TRADES_PARQUET,
    EVALUATE_TEST_PREDICTIONS_CSV,
    EVALUATE_TEST_PREDICTIONS_PARQUET,
)

LOGGER: logging.Logger = logging.getLogger(__name__)


def save_evaluation_outputs(
    predictions: pd.DataFrame,
    trades: pd.DataFrame,
    daily: pd.DataFrame,
    summary: dict[str, float],
) -> None:
    DATA_EVALUATE_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(EVALUATE_TEST_PREDICTIONS_PARQUET, index=False)
    predictions.sample(frac=min(1.0, 0.05), random_state=7).to_csv(
        EVALUATE_TEST_PREDICTIONS_CSV,
        index=False,
    )
    trades.to_parquet(EVALUATE_BACKTEST_TRADES_PARQUET, index=False)
    trades.sample(frac=min(1.0, 0.05), random_state=7).to_csv(
        EVALUATE_BACKTEST_TRADES_CSV,
        index=False,
    )
    daily.to_parquet(EVALUATE_BACKTEST_DAILY_PARQUET, index=False)
    daily.to_csv(EVALUATE_BACKTEST_DAILY_CSV, index=False)
    EVALUATE_BACKTEST_SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    LOGGER.info(
        "Saved evaluate outputs: %s, %s, %s, %s, %s, %s, %s",
        EVALUATE_TEST_PREDICTIONS_PARQUET,
        EVALUATE_TEST_PREDICTIONS_CSV,
        EVALUATE_BACKTEST_TRADES_PARQUET,
        EVALUATE_BACKTEST_TRADES_CSV,
        EVALUATE_BACKTEST_DAILY_PARQUET,
        EVALUATE_BACKTEST_DAILY_CSV,
        EVALUATE_BACKTEST_SUMMARY_JSON,
    )
