from __future__ import annotations

from dataclasses import dataclass
import json
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
    PORTFOLIO_BEST_PARAMS_JSON,
    PORTFOLIO_RISK_COVARIANCE_PARQUET,
    FEATURE_SELECTION_FILTERED_DATASET_PARQUET,
    META_BEST_PARAMS_JSON,
    META_MODEL_JSON,
    XGBOOST_OPTUNA_TRIALS_PARQUET,
)
from core.src.meta_model.evaluate.backtest import (
    ActiveTrade,
    BacktestRuntimeConfig,
    BacktestState,
    PortfolioOptimizerArtifacts,
    XtbCostConfig,
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
from core.src.meta_model.model_contract import DATE_COLUMN, PREDICTION_COLUMN
from core.src.meta_model.portfolio_optimization.alpha_calibration import (
    FittedAlphaCalibrator,
    deserialize_alpha_calibrator,
)
from core.src.meta_model.overfitting import build_overfitting_diagnostics, diagnostics_to_payload
from core.src.meta_model.research_metrics import (
    build_daily_signal_diagnostics,
    summarize_daily_signal_diagnostics,
)
from core.src.meta_model.evaluate.training import iter_model_prediction_days
from core.src.meta_model.evaluate.training import iter_model_prediction_days_frozen_train_only
from core.src.meta_model.meta_labeling.calibration import (
    FittedProbabilityCalibrator,
    deserialize_probability_calibrator,
)
from core.src.meta_model.meta_labeling.features import (
    META_PROBABILITY_COLUMN,
    REFINED_EXPECTED_RETURN_COLUMN,
    REFINED_PREDICTION_COLUMN,
    attach_refined_signal_columns,
    build_primary_context_columns,
)
from core.src.meta_model.meta_labeling.model import (
    MetaModelArtifact,
    deserialize_meta_model_artifact,
    predict_meta_model,
)

LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelEvaluationResult:
    """Immutable result of evaluating one model: predictions, trades, daily PnL, summary."""
    model_name: str
    predictions: pd.DataFrame
    trades: pd.DataFrame
    daily: pd.DataFrame
    allocations: pd.DataFrame
    optimizer_daily: pd.DataFrame
    summary: dict[str, object]


@dataclass(frozen=True)
class EvaluateRuntimeContext:
    """Runtime context: loaded dataset, features, model specs, cost config."""
    dataset: pd.DataFrame
    feature_columns: list[str]
    model_specs: list[ModelSpec]
    unique_test_dates: pd.Index
    xtb_cost_config: XtbCostConfig
    xgboost_trials: pd.DataFrame
    optimizer_artifacts: PortfolioOptimizerArtifacts | None
    optimizer_params: dict[str, object]
    optimizer_alpha_calibrator: FittedAlphaCalibrator | None
    meta_artifact: MetaModelArtifact | None
    meta_probability_calibrator: FittedProbabilityCalibrator | None
    refinement_strategy: str
    soft_shifted_floor: float
    rank_blend_lambda: float


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


def _mapping_float(
    payload: Mapping[str, object],
    key: str,
    default: float,
) -> float:
    raw_value = payload.get(key, default)
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    if isinstance(raw_value, (np.integer, np.floating)):
        return float(raw_value.item())
    if isinstance(raw_value, str):
        try:
            return float(raw_value)
        except ValueError:
            return default
    return default


def _build_backtest_runtime(
    *,
    context: EvaluateRuntimeContext,
    config: BacktestConfig,
    unique_dates: pd.Index,
    cost_config_override: XtbCostConfig | None = None,
) -> BacktestRuntimeConfig:
    candidate_pool_value = context.optimizer_params.get(
        "miqp_candidate_pool_size",
        config.miqp_candidate_pool_size,
    )
    if isinstance(candidate_pool_value, (np.integer, np.floating)):
        candidate_pool_size = int(candidate_pool_value.item())
    elif isinstance(candidate_pool_value, (int, float)):
        candidate_pool_size = int(candidate_pool_value)
    else:
        candidate_pool_size = int(config.miqp_candidate_pool_size)
    return BacktestRuntimeConfig(
        unique_dates=unique_dates,
        top_fraction=config.top_fraction,
        allocation_fraction=config.allocation_fraction,
        action_cap_fraction=config.action_cap_fraction,
        gross_cap_fraction=config.gross_cap_fraction,
        adv_participation_limit=config.adv_participation_limit,
        neutrality_mode=config.neutrality_mode,
        open_hurdle_bps=config.open_hurdle_bps,
        apply_prediction_hurdle=config.apply_prediction_hurdle,
        hold_period_days=config.hold_period_days,
        cost_config=cost_config_override or context.xtb_cost_config,
        portfolio_construction_mode=config.portfolio_construction_mode,
        optimizer_artifacts=context.optimizer_artifacts,
        lambda_risk=_mapping_float(context.optimizer_params, "lambda_risk", config.lambda_risk),
        lambda_turnover=_mapping_float(
            context.optimizer_params,
            "lambda_turnover",
            config.lambda_turnover,
        ),
        lambda_cost=_mapping_float(context.optimizer_params, "lambda_cost", config.lambda_cost),
        max_position_weight=_mapping_float(
            context.optimizer_params,
            "max_position_weight",
            config.max_position_weight,
        ),
        max_sector_weight=_mapping_float(
            context.optimizer_params,
            "max_sector_weight",
            config.max_sector_weight,
        ),
        min_target_weight=_mapping_float(
            context.optimizer_params,
            "min_target_weight",
            config.min_target_weight,
        ),
        no_trade_buffer_bps=_mapping_float(
            context.optimizer_params,
            "no_trade_buffer_bps",
            config.no_trade_buffer_bps,
        ),
        miqp_time_limit_seconds=_mapping_float(
            context.optimizer_params,
            "miqp_time_limit_seconds",
            config.miqp_time_limit_seconds,
        ),
        miqp_relative_gap=_mapping_float(
            context.optimizer_params,
            "miqp_relative_gap",
            config.miqp_relative_gap,
        ),
        miqp_candidate_pool_size=candidate_pool_size,
        miqp_primary_objective_tolerance_bps=_mapping_float(
            context.optimizer_params,
            "miqp_primary_objective_tolerance_bps",
            config.miqp_primary_objective_tolerance_bps,
        ),
    )


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
        if spec.instrument_group == "stock_cash"
        and not spec.symbol.startswith("__default")
    ]
    if not stock_specs:
        return
    minimum_order_value_eur = np.asarray(
        [spec.minimum_order_value_eur for spec in stock_specs],
        dtype=np.float64,
    )
    monthly_free_turnover_eur = np.asarray(
        [spec.monthly_commission_free_turnover_eur for spec in stock_specs],
        dtype=np.float64,
    )
    fx_conversion_bps = np.asarray(
        [spec.fx_conversion_bps for spec in stock_specs],
        dtype=np.float64,
    )
    LOGGER.info(
        "XTB cash-equity cost profile: symbols=%d | minimum_order_value_eur(avg)=%.2f | monthly_free_turnover_eur(avg)=%.2f | fx_conversion_bps(avg)=%.2f",
        len(stock_specs),
        float(minimum_order_value_eur.mean()),
        float(monthly_free_turnover_eur.mean()),
        float(fx_conversion_bps.mean()),
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
    optimizer_artifacts: PortfolioOptimizerArtifacts | None = None
    optimizer_params: dict[str, object] = {}
    optimizer_alpha_calibrator: FittedAlphaCalibrator | None = None
    meta_artifact: MetaModelArtifact | None = None
    meta_probability_calibrator: FittedProbabilityCalibrator | None = None
    if not PORTFOLIO_BEST_PARAMS_JSON.exists() or not PORTFOLIO_RISK_COVARIANCE_PARQUET.exists():
        raise FileNotFoundError(
            "optimizer_miqp mode requires portfolio artifacts. Missing portfolio_best_params.json or risk_covariance.parquet.",
        )
    optimizer_params = json.loads(PORTFOLIO_BEST_PARAMS_JSON.read_text(encoding="utf-8"))
    calibrator_payload = optimizer_params.get("alpha_calibrator")
    if isinstance(calibrator_payload, str) and calibrator_payload.strip():
        optimizer_alpha_calibrator = deserialize_alpha_calibrator(calibrator_payload)
    covariance = pd.read_parquet(PORTFOLIO_RISK_COVARIANCE_PARQUET)
    if "index" in covariance.columns:
        covariance = covariance.set_index("index")
    optimizer_artifacts = PortfolioOptimizerArtifacts(covariance=pd.DataFrame(covariance))
    if not META_BEST_PARAMS_JSON.exists() or not META_MODEL_JSON.exists():
        raise FileNotFoundError(
            "evaluate requires meta_labeling artifacts. Missing meta_best_params.json or meta_model.json.",
        )
    meta_best_params = json.loads(META_BEST_PARAMS_JSON.read_text(encoding="utf-8"))
    meta_model_payload = json.loads(META_MODEL_JSON.read_text(encoding="utf-8"))
    meta_artifact = deserialize_meta_model_artifact(meta_model_payload)
    probability_payload = meta_best_params.get("meta_probability_calibrator")
    if not isinstance(probability_payload, str) or not probability_payload.strip():
        raise ValueError("meta_best_params.json must contain a serialized meta_probability_calibrator.")
    meta_probability_calibrator = deserialize_probability_calibrator(probability_payload)
    return EvaluateRuntimeContext(
        dataset=dataset,
        feature_columns=feature_columns,
        model_specs=model_specs,
        unique_test_dates=_build_unique_test_dates(
            dataset,
            execution_lag_days=config.execution_lag_days,
        ),
        xtb_cost_config=XtbCostConfig(
            account_currency=config.account_currency,
            broker_spec_provider=spec_provider,
        ),
        xgboost_trials=_load_xgboost_trials(),
        optimizer_artifacts=optimizer_artifacts,
        optimizer_params=optimizer_params,
        optimizer_alpha_calibrator=optimizer_alpha_calibrator,
        meta_artifact=meta_artifact,
        meta_probability_calibrator=meta_probability_calibrator,
        refinement_strategy=str(meta_best_params.get("refinement_strategy", "binary_gate")),
        soft_shifted_floor=float(meta_best_params.get("soft_shifted_floor", 0.45)),
        rank_blend_lambda=float(meta_best_params.get("rank_blend_lambda", 0.50)),
    )


def _apply_meta_refinement_to_prediction_day(
    predicted_day: pd.DataFrame,
    *,
    meta_artifact: MetaModelArtifact,
    meta_probability_calibrator: FittedProbabilityCalibrator,
    refinement_strategy: str = "binary_gate",
    soft_shifted_floor: float = 0.45,
    rank_blend_lambda: float = 0.50,
) -> pd.DataFrame:
    enriched = build_primary_context_columns(predicted_day)
    meta_probability = predict_meta_model(meta_artifact, enriched)
    enriched[META_PROBABILITY_COLUMN] = meta_probability_calibrator.transform(
        np.asarray(meta_probability, dtype=np.float64),
    )
    enriched = attach_refined_signal_columns(
        enriched,
        strategy=refinement_strategy,
        soft_shifted_floor=soft_shifted_floor,
        rank_blend_lambda=rank_blend_lambda,
    )
    enriched[PREDICTION_COLUMN] = enriched[REFINED_PREDICTION_COLUMN]
    if REFINED_EXPECTED_RETURN_COLUMN in enriched.columns:
        enriched["expected_return_5d"] = enriched[REFINED_EXPECTED_RETURN_COLUMN]
    return enriched


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
    state = BacktestState(
        initial_equity=config.starting_cash_eur,
        current_equity=config.starting_cash_eur,
        cash_balance=config.starting_cash_eur,
    )
    prediction_parts: list[pd.DataFrame] = []
    runtime = _build_backtest_runtime(
        context=context,
        config=config,
        unique_dates=context.unique_test_dates,
    )
    if config.evaluation_training_mode == "frozen_train_only":
        prediction_day_iter = iter_model_prediction_days_frozen_train_only(
            context.dataset,
            context.feature_columns,
            model_spec,
            execution_lag_days=config.execution_lag_days,
            logger=LOGGER,
        )
    else:
        prediction_day_iter = iter_model_prediction_days(
            context.dataset,
            context.feature_columns,
            model_spec,
            hold_period_days=config.hold_period_days,
            execution_lag_days=config.execution_lag_days,
            logger=LOGGER,
        )
    for predicted_day in prediction_day_iter:
        predicted_day = build_primary_context_columns(predicted_day)
        if context.optimizer_alpha_calibrator is not None:
            expected = context.optimizer_alpha_calibrator.transform(
                pd.to_numeric(predicted_day["prediction"], errors="coerce").to_numpy(dtype=np.float64),
            )
            predicted_day = pd.DataFrame(predicted_day.copy())
            predicted_day["expected_return_5d"] = expected
        if context.meta_artifact is not None and context.meta_probability_calibrator is not None:
            predicted_day = _apply_meta_refinement_to_prediction_day(
                predicted_day,
                meta_artifact=context.meta_artifact,
                meta_probability_calibrator=context.meta_probability_calibrator,
                refinement_strategy=context.refinement_strategy,
                soft_shifted_floor=context.soft_shifted_floor,
                rank_blend_lambda=context.rank_blend_lambda,
            )
        prediction_parts.append(predicted_day)
        process_prediction_day(state, predicted_day, runtime, logger=LOGGER)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions["model_name"] = model_spec.model_name
    trades, daily, backtest_summary = finalize_backtest_state(state)
    allocations = pd.DataFrame(state.allocation_rows)
    optimizer_daily = pd.DataFrame(state.optimizer_daily_rows)
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
        allocations=allocations,
        optimizer_daily=optimizer_daily,
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
    context: EvaluateRuntimeContext,
    config: BacktestConfig,
    xtb_cost_config: XtbCostConfig,
) -> list[ActiveTrade]:
    selected_predictions = pd.DataFrame(selected_result.predictions)
    latest_prediction_date = pd.Timestamp(selected_predictions[DATE_COLUMN].max())
    latest_predictions = pd.DataFrame(
        selected_predictions.loc[selected_predictions[DATE_COLUMN] == latest_prediction_date].copy(),
    )
    active_state = BacktestState(
        initial_equity=_summary_float(selected_result.summary, "final_equity"),
        current_equity=_summary_float(selected_result.summary, "final_equity"),
        cash_balance=_summary_float(selected_result.summary, "final_equity"),
    )
    runtime = _build_backtest_runtime(
        context=context,
        config=config,
        unique_dates=pd.Index([latest_prediction_date]),
        cost_config_override=xtb_cost_config,
    )
    process_prediction_day(active_state, latest_predictions, runtime)
    return list(active_state.active_trades)


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
    """Run the full evaluate pipeline: train, backtest, rank, and save outputs."""
    config = backtest_config or BacktestConfig()
    context = _build_runtime_context(config, dataset_path)
    model_results, leaderboard = _evaluate_models(context=context, config=config)
    promoted_model_name = _select_promoted_model(leaderboard)
    selected_result = next(
        result for result in model_results if result.model_name == promoted_model_name
    )
    manual_trades = _build_manual_trades(
        selected_result,
        context=context,
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
        portfolio_target_allocations=selected_result.allocations,
        portfolio_optimizer_daily=selected_result.optimizer_daily,
        portfolio_optimizer_summary={
            "selected_model_name": promoted_model_name,
            "row_count_allocations": int(len(selected_result.allocations)),
            "row_count_optimizer_daily": int(len(selected_result.optimizer_daily)),
        },
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
    """Entry point for the evaluate pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_evaluate_pipeline()


if __name__ == "__main__":
    main()
