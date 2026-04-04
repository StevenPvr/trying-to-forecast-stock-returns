from __future__ import annotations

from dataclasses import dataclass
import logging
import sys
from pathlib import Path
from typing import Mapping

PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from core.src.meta_model.broker_xtb.bridge import (
    build_manual_execution_bundle,
    save_manual_execution_bundle,
)
from core.src.meta_model.broker_xtb.specs import (
    BrokerSpecProvider,
    build_default_spec_provider,
    save_broker_snapshots,
)
from core.src.meta_model.broker_xtb.universe import build_tradable_universe, save_tradable_universe
from core.src.meta_model.data.paths import (
    FEATURE_SELECTION_FILTERED_DATASET_PARQUET,
    XGBOOST_OPTUNA_TRIALS_PARQUET,
)
from core.src.meta_model.evaluate.backtest import (
    ActiveTrade,
    BacktestState,
    XtbCostConfig,
    allocate_signal_candidates,
    build_daily_signal_candidates,
    finalize_backtest_state,
    process_prediction_day,
)
from core.src.meta_model.evaluate.config import BacktestConfig, TARGET_COLUMN
from core.src.meta_model.evaluate.dataset import (
    build_feature_columns,
    load_preprocessed_evaluation_dataset,
    validate_feature_schema_manifest,
)
from core.src.meta_model.evaluate.io import save_evaluation_outputs
from core.src.meta_model.evaluate.parameters import load_selected_xgboost_configuration
from core.src.meta_model.model_registry.main import ModelSpec, build_default_model_specs
from core.src.meta_model.model_contract import DATE_COLUMN
from core.src.meta_model.overfitting import build_overfitting_diagnostics, diagnostics_to_payload
from core.src.meta_model.research_metrics import (
    build_daily_signal_diagnostics,
    summarize_daily_signal_diagnostics,
)
from core.src.meta_model.evaluate.training import iter_model_prediction_days

LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelEvaluationResult:
    model_name: str
    predictions: pd.DataFrame
    trades: pd.DataFrame
    daily: pd.DataFrame
    summary: dict[str, object]


@dataclass(frozen=True)
class EvaluateRuntimeContext:
    dataset: pd.DataFrame
    feature_columns: list[str]
    model_specs: list[ModelSpec]
    unique_test_dates: pd.Index
    xtb_cost_config: XtbCostConfig
    xgboost_trials: pd.DataFrame


def _float_payload(payload: dict[str, float]) -> dict[str, object]:
    return {key: float(value) for key, value in payload.items()}


def _summary_payload(summary: Mapping[str, float]) -> dict[str, object]:
    return {key: float(value) for key, value in summary.items()}


def _leaderboard_records(leaderboard: pd.DataFrame) -> list[dict[str, object]]:
    raw_records = leaderboard.to_dict(orient="records")
    return [{str(key): value for key, value in record.items()} for record in raw_records]


def _build_overfitting_report(
    promoted_model_name: str,
    deployable_model_name: str,
    leaderboard_records: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "selected_model_name": promoted_model_name,
        "deployable_model_name": deployable_model_name,
        "leaderboard": leaderboard_records,
    }


def _summary_float(summary: dict[str, object], key: str) -> float:
    return float(np.asarray(summary[key], dtype=np.float64).item())


def _build_model_leaderboard_row(
    model_name: str,
    predictions: pd.DataFrame,
    trades: pd.DataFrame,
    daily: pd.DataFrame,
    summary: Mapping[str, float],
) -> dict[str, object]:
    daily_signal_diagnostics = build_daily_signal_diagnostics(
        predictions,
        target_column=TARGET_COLUMN,
    )
    row: dict[str, object] = {"model_name": model_name}
    row.update(_summary_payload(summary))
    row.update(_float_payload(summarize_daily_signal_diagnostics(daily_signal_diagnostics)))
    row["prediction_day_count"] = float(
        len(pd.Index(pd.to_datetime(predictions[DATE_COLUMN]).drop_duplicates())),
    )
    row["trade_count"] = float(len(trades))
    row["daily_row_count"] = float(len(daily))
    return row


def _select_promoted_model(leaderboard: pd.DataFrame) -> str:
    deployable = pd.DataFrame(leaderboard.loc[leaderboard["is_deployable"]].copy())
    ranked_source = deployable if not deployable.empty else leaderboard
    ranked = ranked_source.sort_values(
        [
            "is_deployable",
            "alpha_over_benchmark_net",
            "daily_rank_ic_ir",
            "calmar_ratio",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return str(ranked.loc[0, "model_name"])


def _select_evaluate_model_specs(model_specs: list[ModelSpec]) -> list[ModelSpec]:
    selected_model_specs = [
        model_spec
        for model_spec in model_specs
        if model_spec.model_name == "xgboost"
    ]
    if not selected_model_specs:
        raise ValueError("Evaluate pipeline requires an xgboost model spec.")
    return selected_model_specs


def _build_unique_test_dates(
    dataset: pd.DataFrame,
    *,
    execution_lag_days: int,
) -> pd.Index:
    raw_test_dates = pd.Index(
        pd.to_datetime(
            dataset.loc[dataset["dataset_split"] == "test", "date"],
        ).drop_duplicates().sort_values(),
    )
    return pd.Index(raw_test_dates.tolist()[execution_lag_days:])


def _resolve_latest_trade_date(dataset: pd.DataFrame) -> pd.Timestamp:
    test_dates = pd.to_datetime(dataset.loc[dataset["dataset_split"] == "test", "date"])
    if len(test_dates) == 0:
        return pd.Timestamp("1970-01-01")
    return pd.Timestamp(pd.Index(test_dates).max())


def _load_xgboost_trials() -> pd.DataFrame:
    if XGBOOST_OPTUNA_TRIALS_PARQUET.exists():
        return pd.read_parquet(XGBOOST_OPTUNA_TRIALS_PARQUET)
    return pd.DataFrame()


def _log_cost_profile(spec_provider: BrokerSpecProvider) -> None:
    stock_specs = [
        spec
        for spec in spec_provider.specs
        if spec.instrument_group == "stock_cfd"
        and not spec.symbol.startswith("__default")
    ]
    if not stock_specs:
        return
    spread_bps = np.asarray([spec.spread_bps for spec in stock_specs], dtype=np.float64)
    slippage_bps = np.asarray([spec.slippage_bps for spec in stock_specs], dtype=np.float64)
    long_swap_bps_daily = np.asarray(
        [spec.long_swap_bps_daily for spec in stock_specs],
        dtype=np.float64,
    )
    short_swap_bps_daily = np.asarray(
        [spec.short_swap_bps_daily for spec in stock_specs],
        dtype=np.float64,
    )
    LOGGER.info(
        "XTB equity cost profile (specs; commission-free cash assumption): symbols=%d | spread_bps(avg/min/max)=%.3f/%.3f/%.3f | slippage_bps(avg/min/max)=%.3f/%.3f/%.3f | long_swap_bps_daily(avg)=%.3f | short_swap_bps_daily(avg)=%.3f",
        len(stock_specs),
        float(spread_bps.mean()),
        float(spread_bps.min()),
        float(spread_bps.max()),
        float(slippage_bps.mean()),
        float(slippage_bps.min()),
        float(slippage_bps.max()),
        float(long_swap_bps_daily.mean()),
        float(short_swap_bps_daily.mean()),
    )


def _build_runtime_context(
    config: BacktestConfig,
    dataset_path: Path,
) -> EvaluateRuntimeContext:
    dataset = load_preprocessed_evaluation_dataset(dataset_path)
    feature_columns = build_feature_columns(dataset)
    validate_feature_schema_manifest(feature_columns, dataset_path)
    selected_configuration = load_selected_xgboost_configuration()
    default_model_specs = build_default_model_specs(
        xgboost_params=selected_configuration.params,
        xgboost_training_rounds=selected_configuration.training_rounds,
    )
    model_specs = _select_evaluate_model_specs(default_model_specs)
    LOGGER.info(
        "Evaluate pipeline started: rows=%d | features=%d | best_trial=%d | training_rounds=%d | models=%s",
        len(dataset),
        len(feature_columns),
        selected_configuration.selected_trial_number,
        selected_configuration.training_rounds,
        ",".join(spec.model_name for spec in model_specs),
    )
    spec_provider = build_default_spec_provider()
    save_broker_snapshots(spec_provider)
    _log_cost_profile(spec_provider)
    tradable_universe = build_tradable_universe(
        dataset,
        spec_provider,
        trade_date=_resolve_latest_trade_date(dataset),
        max_spread_bps=config.max_spread_bps,
    )
    save_tradable_universe(tradable_universe)
    return EvaluateRuntimeContext(
        dataset=dataset,
        feature_columns=feature_columns,
        model_specs=model_specs,
        unique_test_dates=_build_unique_test_dates(
            dataset,
            execution_lag_days=config.execution_lag_days,
        ),
        xtb_cost_config=XtbCostConfig(broker_spec_provider=spec_provider),
        xgboost_trials=_load_xgboost_trials(),
    )


def _build_overfitting_diagnostics_payload(
    model_name: str,
    daily: pd.DataFrame,
    xgboost_trials: pd.DataFrame,
) -> dict[str, object]:
    is_xgboost_model = model_name == "xgboost" and not xgboost_trials.empty
    overfitting_diagnostics = build_overfitting_diagnostics(
        pd.Series(daily["realized_return"], dtype=float) if not daily.empty else pd.Series(dtype=float),
        trial_count=len(xgboost_trials) if is_xgboost_model else 1,
        trials_frame=xgboost_trials if is_xgboost_model else None,
    )
    return _float_payload(diagnostics_to_payload(overfitting_diagnostics))


def _is_deployable_summary_row(
    summary_row: dict[str, object],
    config: BacktestConfig,
) -> bool:
    return bool(
        _summary_float(summary_row, "alpha_over_benchmark_net") > 0.0
        and _summary_float(summary_row, "daily_rank_ic_ir") > 0.0
        and _summary_float(summary_row, "pbo") <= config.pbo_max_threshold
        and _summary_float(summary_row, "deflated_sharpe_ratio") >= config.dsr_min_threshold
    )


def _evaluate_single_model(
    model_spec: ModelSpec,
    *,
    context: EvaluateRuntimeContext,
    config: BacktestConfig,
) -> ModelEvaluationResult:
    state = BacktestState()
    prediction_parts: list[pd.DataFrame] = []
    for predicted_day in iter_model_prediction_days(
        context.dataset,
        context.feature_columns,
        model_spec,
        hold_period_days=config.hold_period_days,
        execution_lag_days=config.execution_lag_days,
        logger=LOGGER,
    ):
        prediction_parts.append(predicted_day)
        process_prediction_day(
            state=state,
            daily_predictions=predicted_day,
            unique_dates=context.unique_test_dates,
            top_fraction=config.top_fraction,
            allocation_fraction=config.allocation_fraction,
            action_cap_fraction=config.action_cap_fraction,
            gross_cap_fraction=config.gross_cap_fraction,
            adv_participation_limit=config.adv_participation_limit,
            neutrality_mode=config.neutrality_mode,
            open_hurdle_bps=config.open_hurdle_bps,
            apply_prediction_hurdle=config.apply_prediction_hurdle,
            hold_period_days=config.hold_period_days,
            cost_config=context.xtb_cost_config,
            logger=LOGGER,
        )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions["model_name"] = model_spec.model_name
    trades, daily, backtest_summary = finalize_backtest_state(state)
    if not trades.empty:
        trades["model_name"] = model_spec.model_name
    if not daily.empty:
        daily["model_name"] = model_spec.model_name
    summary_row = _build_model_leaderboard_row(
        model_spec.model_name,
        predictions,
        trades,
        daily,
        backtest_summary,
    )
    summary_row.update(
        _build_overfitting_diagnostics_payload(
            model_spec.model_name,
            daily,
            context.xgboost_trials,
        ),
    )
    summary_row["is_deployable"] = _is_deployable_summary_row(summary_row, config)
    return ModelEvaluationResult(
        model_name=model_spec.model_name,
        predictions=predictions,
        trades=trades,
        daily=daily,
        summary=summary_row,
    )


def _evaluate_models(
    *,
    context: EvaluateRuntimeContext,
    config: BacktestConfig,
) -> tuple[list[ModelEvaluationResult], pd.DataFrame]:
    model_results = [
        _evaluate_single_model(
            model_spec,
            context=context,
            config=config,
        )
        for model_spec in context.model_specs
    ]
    leaderboard_rows = [result.summary for result in model_results]
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["alpha_over_benchmark_net", "daily_rank_ic_ir", "sharpe_ratio"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return model_results, leaderboard


def _build_manual_trades(
    selected_result: ModelEvaluationResult,
    *,
    config: BacktestConfig,
    xtb_cost_config: XtbCostConfig,
) -> list[ActiveTrade]:
    selected_predictions = pd.DataFrame(selected_result.predictions)
    latest_prediction_date = pd.Timestamp(selected_predictions[DATE_COLUMN].max())
    latest_predictions = pd.DataFrame(
        selected_predictions.loc[selected_predictions[DATE_COLUMN] == latest_prediction_date].copy(),
    )
    manual_candidates = build_daily_signal_candidates(
        latest_predictions,
        top_fraction=config.top_fraction,
        cost_config=xtb_cost_config,
        expected_holding_days=config.hold_period_days,
        neutrality_mode=config.neutrality_mode,
    )
    return allocate_signal_candidates(
        trade_date=latest_prediction_date,
        candidates=manual_candidates,
        active_trades=[],
        current_equity=_summary_float(selected_result.summary, "final_equity"),
        hold_period_days=config.hold_period_days,
        allocation_fraction=config.allocation_fraction,
        action_cap_fraction=config.action_cap_fraction,
        gross_cap_fraction=config.gross_cap_fraction,
        adv_participation_limit=config.adv_participation_limit,
        neutrality_mode=config.neutrality_mode,
        open_hurdle_bps=config.open_hurdle_bps,
        apply_prediction_hurdle=config.apply_prediction_hurdle,
        unique_dates=pd.Index([latest_prediction_date]),
    )


def _build_pipeline_summary(
    selected_result: ModelEvaluationResult,
    *,
    promoted_model_name: str,
    leaderboard: pd.DataFrame,
) -> tuple[dict[str, object], dict[str, object]]:
    leaderboard_records = _leaderboard_records(leaderboard)
    deployable_model_name = (
        promoted_model_name if bool(selected_result.summary["is_deployable"]) else "none"
    )
    summary: dict[str, object] = {
        **dict(selected_result.summary),
        "selected_model_name": promoted_model_name,
        "deployable_model_name": deployable_model_name,
        "leaderboard": leaderboard_records,
    }
    overfitting_report = _build_overfitting_report(
        promoted_model_name=promoted_model_name,
        deployable_model_name=deployable_model_name,
        leaderboard_records=leaderboard_records,
    )
    return summary, overfitting_report


def run_evaluate_pipeline(
    backtest_config: BacktestConfig | None = None,
    dataset_path: Path = FEATURE_SELECTION_FILTERED_DATASET_PARQUET,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    config = backtest_config or BacktestConfig()
    context = _build_runtime_context(config, dataset_path)
    model_results, leaderboard = _evaluate_models(context=context, config=config)
    promoted_model_name = _select_promoted_model(leaderboard)
    selected_result = next(
        result for result in model_results if result.model_name == promoted_model_name
    )
    manual_trades = _build_manual_trades(
        selected_result,
        config=config,
        xtb_cost_config=context.xtb_cost_config,
    )
    manual_orders, manual_watchlist, execution_checklist = build_manual_execution_bundle(manual_trades)
    save_manual_execution_bundle(manual_orders, manual_watchlist, execution_checklist)
    summary, overfitting_report = _build_pipeline_summary(
        selected_result,
        promoted_model_name=promoted_model_name,
        leaderboard=leaderboard,
    )
    save_evaluation_outputs(
        selected_result.predictions,
        selected_result.trades,
        selected_result.daily,
        summary,
        leaderboard=leaderboard,
        overfitting_report=overfitting_report,
    )
    LOGGER.info(
        "Evaluate pipeline completed: selected_model=%s | trades=%d | final_equity=%.6f | total_return=%.6f | sharpe=%.6f",
        promoted_model_name,
        len(selected_result.trades),
        _summary_float(summary, "final_equity"),
        _summary_float(summary, "total_return"),
        _summary_float(summary, "sharpe_ratio"),
    )
    return (
        selected_result.trades,
        selected_result.daily,
        summary,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_evaluate_pipeline()


if __name__ == "__main__":
    main()
