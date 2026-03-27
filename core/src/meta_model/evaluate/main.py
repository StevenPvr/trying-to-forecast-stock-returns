from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from core.src.meta_model.evaluate.backtest import (
    BacktestState,
    XtbCostConfig,
    finalize_backtest_state,
    process_prediction_day,
)
from core.src.meta_model.evaluate.config import BacktestConfig
from core.src.meta_model.evaluate.dataset import (
    build_feature_columns,
    load_preprocessed_evaluation_dataset,
)
from core.src.meta_model.evaluate.io import save_evaluation_outputs
from core.src.meta_model.evaluate.parameters import load_selected_xgboost_configuration
from core.src.meta_model.evaluate.training import iter_walk_forward_prediction_days

LOGGER: logging.Logger = logging.getLogger(__name__)


def run_evaluate_pipeline(
    backtest_config: BacktestConfig | None = None,
) -> tuple[object, object, dict[str, float]]:
    config = backtest_config or BacktestConfig()
    dataset = load_preprocessed_evaluation_dataset()
    feature_columns = build_feature_columns(dataset)
    selected_configuration = load_selected_xgboost_configuration()
    LOGGER.info(
        "Evaluate pipeline started: rows=%d | features=%d | selected_trial=%d | training_rounds=%d | mode=walk_forward_daily_retrain",
        len(dataset),
        len(feature_columns),
        selected_configuration.selected_trial_number,
        selected_configuration.training_rounds,
    )
    state = BacktestState()
    prediction_parts: list[pd.DataFrame] = []
    unique_test_dates = pd.Index(
        pd.to_datetime(
            dataset.loc[dataset["dataset_split"] == "test", "date"],
        ).drop_duplicates().sort_values(),
    )
    xtb_cost_config = XtbCostConfig(
        transaction_cost_rate_per_side=config.transaction_cost_rate_per_side,
        long_daily_financing_rate=config.long_daily_financing_rate,
        short_daily_financing_rate=config.short_daily_financing_rate,
    )
    for predicted_day in iter_walk_forward_prediction_days(
        dataset,
        feature_columns,
        selected_configuration.params,
        num_boost_round=selected_configuration.training_rounds,
        hold_period_days=config.hold_period_days,
        logger=LOGGER,
    ):
        prediction_parts.append(predicted_day)
        process_prediction_day(
            state=state,
            daily_predictions=predicted_day,
            unique_dates=unique_test_dates,
            top_fraction=config.top_fraction,
            allocation_fraction=config.allocation_fraction,
            action_cap_fraction=config.action_cap_fraction,
            gross_cap_fraction=config.gross_cap_fraction,
            hold_period_days=config.hold_period_days,
            cost_config=xtb_cost_config,
            logger=LOGGER,
        )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    trades, daily, summary = finalize_backtest_state(state)
    save_evaluation_outputs(predictions, trades, daily, summary)
    LOGGER.info(
        "Evaluate pipeline completed: trades=%d | final_equity=%.6f | total_return=%.6f | sharpe=%.6f",
        len(trades),
        summary["final_equity"],
        summary["total_return"],
        summary["sharpe_ratio"],
    )
    return trades, daily, summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_evaluate_pipeline()


if __name__ == "__main__":
    main()
