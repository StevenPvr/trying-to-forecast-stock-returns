from __future__ import annotations

import json
import logging


import pandas as pd

from core.src.meta_model.data.paths import (
    DATA_EVALUATE_DIR,
    EVALUATE_BACKTEST_DAILY_CSV,
    EVALUATE_BACKTEST_DAILY_PARQUET,
    EVALUATE_EXECUTION_CHECKLIST_JSON,
    EVALUATE_MANUAL_ORDERS_CSV,
    EVALUATE_MANUAL_WATCHLIST_CSV,
    EVALUATE_MODEL_LEADERBOARD_JSON,
    EVALUATE_OVERFITTING_REPORT_JSON,
    EVALUATE_POST_TRADE_RECONCILIATION_PARQUET,
    EVALUATE_PORTFOLIO_OPTIMIZER_DAILY_PARQUET,
    EVALUATE_PORTFOLIO_OPTIMIZER_SUMMARY_JSON,
    EVALUATE_PORTFOLIO_TARGET_ALLOCATIONS_PARQUET,
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
    summary: dict[str, object],
    leaderboard: pd.DataFrame | None = None,
    overfitting_report: dict[str, object] | None = None,
    portfolio_target_allocations: pd.DataFrame | None = None,
    portfolio_optimizer_daily: pd.DataFrame | None = None,
    portfolio_optimizer_summary: dict[str, object] | None = None,
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
    if leaderboard is not None:
        EVALUATE_MODEL_LEADERBOARD_JSON.write_text(
            leaderboard.to_json(orient="records", indent=2),
            encoding="utf-8",
        )
    if overfitting_report is not None:
        EVALUATE_OVERFITTING_REPORT_JSON.write_text(
            json.dumps(overfitting_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if portfolio_target_allocations is not None:
        portfolio_target_allocations.to_parquet(EVALUATE_PORTFOLIO_TARGET_ALLOCATIONS_PARQUET, index=False)
    if portfolio_optimizer_daily is not None:
        portfolio_optimizer_daily.to_parquet(EVALUATE_PORTFOLIO_OPTIMIZER_DAILY_PARQUET, index=False)
    if portfolio_optimizer_summary is not None:
        EVALUATE_PORTFOLIO_OPTIMIZER_SUMMARY_JSON.write_text(
            json.dumps(portfolio_optimizer_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    LOGGER.info(
        "Saved evaluate outputs: %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s",
        EVALUATE_TEST_PREDICTIONS_PARQUET,
        EVALUATE_TEST_PREDICTIONS_CSV,
        EVALUATE_BACKTEST_TRADES_PARQUET,
        EVALUATE_BACKTEST_TRADES_CSV,
        EVALUATE_BACKTEST_DAILY_PARQUET,
        EVALUATE_BACKTEST_DAILY_CSV,
        EVALUATE_BACKTEST_SUMMARY_JSON,
        EVALUATE_MODEL_LEADERBOARD_JSON,
        EVALUATE_MANUAL_ORDERS_CSV,
        EVALUATE_MANUAL_WATCHLIST_CSV,
        EVALUATE_EXECUTION_CHECKLIST_JSON,
        EVALUATE_POST_TRADE_RECONCILIATION_PARQUET,
        EVALUATE_PORTFOLIO_TARGET_ALLOCATIONS_PARQUET,
        EVALUATE_PORTFOLIO_OPTIMIZER_DAILY_PARQUET,
        EVALUATE_PORTFOLIO_OPTIMIZER_SUMMARY_JSON,
    )
