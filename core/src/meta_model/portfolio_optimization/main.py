from __future__ import annotations

import json
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
import logging
import multiprocessing as mp
from multiprocessing.context import BaseContext
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.paths import FEATURE_SELECTION_FILTERED_DATASET_PARQUET
from core.src.meta_model.data.paths import (
    META_BEST_PARAMS_JSON,
    META_TRAIN_OOF_PREDICTIONS_PARQUET,
    META_VAL_PREDICTIONS_PARQUET,
)
from core.src.meta_model.evaluate.backtest import (
    BacktestProgressConfig,
    BacktestRunConfig,
    BacktestRuntimeConfig,
    EXPECTED_RETURN_COLUMN,
    PortfolioOptimizerArtifacts,
    XtbCostConfig,
    run_signal_backtest_with_diagnostics,
)
from core.src.meta_model.evaluate.config import BacktestConfig
from core.src.meta_model.model_contract import (
    DATE_COLUMN,
    PREDICTION_COLUMN,
    SPLIT_COLUMN,
    TICKER_COLUMN,
    VAL_SPLIT_NAME,
)
from core.src.meta_model.meta_labeling.features import (
    META_PROBABILITY_COLUMN,
    PRIMARY_PREDICTION_COLUMN,
    REFINED_EXPECTED_RETURN_COLUMN,
    attach_refined_signal_columns,
)
from core.src.meta_model.portfolio_optimization.alpha_calibration import (
    FittedAlphaCalibrator,
    fit_alpha_calibrator_train_only,
    serialize_alpha_calibrator,
)
from core.src.meta_model.portfolio_optimization.config import PortfolioOptimizationConfig
from core.src.meta_model.portfolio_optimization.io import save_portfolio_optimization_outputs
from core.src.meta_model.portfolio_optimization.risk_model import fit_train_only_covariance_model
from core.src.meta_model.runtime_parallelism import resolve_executor_worker_count
from core.src.meta_model.split_guard import assert_train_only_fit_frame

LOGGER: logging.Logger = logging.getLogger(__name__)

_trial_worker_predictions: pd.DataFrame | None = None
_trial_worker_runtime: PortfolioOptimizationConfig | None = None
_trial_worker_config: BacktestConfig | None = None
_trial_worker_covariance: dict[int, pd.DataFrame] | None = None


@dataclass(frozen=True)
class PortfolioTrialTask:
    trial_index: int
    params: dict[str, float | int]


@dataclass(frozen=True)
class PortfolioTrialResult:
    trial_index: int
    state: str
    params: dict[str, float | int]
    objective_score: float
    metrics: dict[str, float]
    daily_row_count: int
    allocation_row_count: int
    elapsed_seconds: float
    error_message: str | None = None


def _init_trial_worker(
    predictions: pd.DataFrame,
    runtime: PortfolioOptimizationConfig,
    config: BacktestConfig,
    covariance_by_lookback: dict[int, pd.DataFrame],
) -> None:
    global _trial_worker_predictions, _trial_worker_runtime, _trial_worker_config, _trial_worker_covariance
    _trial_worker_predictions = predictions
    _trial_worker_runtime = runtime
    _trial_worker_config = config
    _trial_worker_covariance = covariance_by_lookback


def _run_grid_trial_worker(task: PortfolioTrialTask) -> PortfolioTrialResult:
    if (
        _trial_worker_predictions is None
        or _trial_worker_runtime is None
        or _trial_worker_config is None
        or _trial_worker_covariance is None
    ):
        raise RuntimeError("portfolio trial worker not initialized.")
    return _run_grid_trial(
        task,
        predictions=_trial_worker_predictions,
        runtime=_trial_worker_runtime,
        config=_trial_worker_config,
        covariance_by_lookback=_trial_worker_covariance,
    )


def _format_duration(seconds: float) -> str:
    rounded_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(rounded_seconds, 60)
    hours, mins = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{mins:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _score_trial(
    predictions: pd.DataFrame,
    *,
    config: BacktestConfig,
    runtime: PortfolioOptimizationConfig,
    risk_covariance: pd.DataFrame,
    lambda_risk: float,
    lambda_turnover: float,
    lambda_cost: float,
    max_position_weight: float,
    max_sector_weight: float,
    no_trade_buffer_bps: float,
    logger: logging.Logger | None = None,
    progress_label: str | None = None,
    progress_log_every: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], pd.DataFrame, pd.DataFrame]:
    run_config = BacktestRunConfig(
        runtime=BacktestRuntimeConfig(
            unique_dates=pd.Index([]),
            top_fraction=config.top_fraction,
            allocation_fraction=config.allocation_fraction,
            action_cap_fraction=config.action_cap_fraction,
            gross_cap_fraction=config.gross_cap_fraction,
            adv_participation_limit=config.adv_participation_limit,
            neutrality_mode=config.neutrality_mode,
            open_hurdle_bps=config.open_hurdle_bps,
            apply_prediction_hurdle=config.apply_prediction_hurdle,
            hold_period_days=config.hold_period_days,
            cost_config=XtbCostConfig(account_currency=config.account_currency),
            portfolio_construction_mode="optimizer_miqp",
            optimizer_artifacts=PortfolioOptimizerArtifacts(covariance=risk_covariance),
            lambda_risk=lambda_risk,
            lambda_turnover=lambda_turnover,
            lambda_cost=lambda_cost,
            max_position_weight=max_position_weight,
            max_sector_weight=max_sector_weight,
            min_target_weight=config.min_target_weight,
            no_trade_buffer_bps=no_trade_buffer_bps,
            miqp_time_limit_seconds=runtime.miqp_time_limit_seconds,
            miqp_relative_gap=runtime.miqp_relative_gap,
            miqp_candidate_pool_size=runtime.miqp_candidate_pool_size,
            miqp_primary_objective_tolerance_bps=runtime.miqp_primary_objective_tolerance_bps,
        ),
        starting_cash_eur=config.starting_cash_eur,
    )
    return run_signal_backtest_with_diagnostics(
        predictions,
        run_config,
        progress=BacktestProgressConfig(
            logger=logger,
            progress_label=progress_label,
            progress_log_every=progress_log_every,
        ),
    )


def _score_summary_for_optimization(summary: dict[str, float]) -> float:
    sharpe_ratio = float(summary.get("sharpe_ratio", 0.0))
    alpha_over_benchmark = float(summary.get("alpha_over_benchmark_net", 0.0))
    calmar_ratio = float(summary.get("calmar_ratio", 0.0))
    max_drawdown = float(summary.get("max_drawdown", 0.0))
    turnover_annualized = float(summary.get("turnover_annualized", 0.0))
    score = sharpe_ratio + 0.05 * alpha_over_benchmark + 0.01 * calmar_ratio
    if alpha_over_benchmark <= 0.0:
        score -= 5.0 + abs(alpha_over_benchmark)
    if max_drawdown < -0.15:
        score -= 100.0 * abs(max_drawdown + 0.15)
    if turnover_annualized > 25.0:
        score -= 0.1 * (turnover_annualized - 25.0)
    return float(score)


def _fit_risk_model_cache(
    train: pd.DataFrame,
    *,
    runtime: PortfolioOptimizationConfig,
) -> dict[int, pd.DataFrame]:
    cache: dict[int, pd.DataFrame] = {}
    lookback_choices = sorted(
        {*runtime.covariance_lookback_days_grid, runtime.covariance_lookback_days},
    )
    for lookback_days in lookback_choices:
        risk_model = fit_train_only_covariance_model(
            train,
            lookback_days=lookback_days,
            min_history_days=runtime.covariance_min_history_days,
        )
        cache[int(lookback_days)] = risk_model.covariance
        LOGGER.info(
            "Portfolio risk model fit: covariance_shape=%s | lookback_days=%d",
            risk_model.covariance.shape,
            int(lookback_days),
        )
    return cache


def _build_trial_grid(
    runtime: PortfolioOptimizationConfig,
) -> list[PortfolioTrialTask]:
    tasks: list[PortfolioTrialTask] = []
    for trial_index, values in enumerate(
        product(
            runtime.lambda_risk_grid,
            runtime.lambda_turnover_grid,
            runtime.no_trade_buffer_bps_grid,
            runtime.max_position_weight_grid,
            runtime.max_sector_weight_grid,
            runtime.covariance_lookback_days_grid,
        ),
    ):
        (
            lambda_risk,
            lambda_turnover,
            no_trade_buffer_bps,
            max_position_weight,
            max_sector_weight,
            covariance_lookback_days,
        ) = values
        tasks.append(
            PortfolioTrialTask(
                trial_index=trial_index,
                params={
                    "lambda_risk": float(lambda_risk),
                    "lambda_turnover": float(lambda_turnover),
                    "no_trade_buffer_bps": float(no_trade_buffer_bps),
                    "max_position_weight": float(max_position_weight),
                    "max_sector_weight": float(max_sector_weight),
                    "covariance_lookback_days": int(covariance_lookback_days),
                },
            ),
        )
    return tasks


def _resolve_trial_process_context() -> BaseContext:
    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context()


def _run_grid_trial(
    task: PortfolioTrialTask,
    *,
    predictions: pd.DataFrame,
    runtime: PortfolioOptimizationConfig,
    config: BacktestConfig,
    covariance_by_lookback: dict[int, pd.DataFrame],
) -> PortfolioTrialResult:
    started_at = time.perf_counter()
    params = task.params
    LOGGER.info(
        "Portfolio grid trial %d started | lambda_risk=%.4f | lambda_turnover=%.4f | max_position_weight=%.4f | max_sector_weight=%.4f | no_trade_buffer_bps=%.2f | covariance_lookback_days=%d",
        task.trial_index,
        float(params["lambda_risk"]),
        float(params["lambda_turnover"]),
        float(params["max_position_weight"]),
        float(params["max_sector_weight"]),
        float(params["no_trade_buffer_bps"]),
        int(params["covariance_lookback_days"]),
    )
    try:
        _, train_daily, train_summary, train_allocations, _ = _score_trial(
            predictions,
            config=config,
            runtime=runtime,
            risk_covariance=covariance_by_lookback[int(params["covariance_lookback_days"])],
            lambda_risk=float(params["lambda_risk"]),
            lambda_turnover=float(params["lambda_turnover"]),
            lambda_cost=runtime.lambda_cost,
            max_position_weight=float(params["max_position_weight"]),
            max_sector_weight=float(params["max_sector_weight"]),
            no_trade_buffer_bps=float(params["no_trade_buffer_bps"]),
            logger=LOGGER,
            progress_label=f"Portfolio grid trial {task.trial_index} backtest",
            progress_log_every=runtime.trial_progress_log_every_days,
        )
        score = _score_summary_for_optimization(train_summary)
        metrics = {
            "sharpe_ratio": float(train_summary.get("sharpe_ratio", 0.0)),
            "alpha_over_benchmark_net": float(
                train_summary.get("alpha_over_benchmark_net", 0.0),
            ),
            "calmar_ratio": float(train_summary.get("calmar_ratio", 0.0)),
            "max_drawdown": float(train_summary.get("max_drawdown", 0.0)),
            "turnover_annualized": float(train_summary.get("turnover_annualized", 0.0)),
            "all_cash": float(1.0 if len(train_allocations) == 0 else 0.0),
        }
        LOGGER.info(
            "Portfolio grid trial %d completed | score=%.6f | sharpe=%.6f | alpha=%.6f | drawdown=%.6f | turnover=%.6f | allocations=%d | all_cash=%s | elapsed=%s",
            task.trial_index,
            score,
            metrics["sharpe_ratio"],
            metrics["alpha_over_benchmark_net"],
            metrics["max_drawdown"],
            metrics["turnover_annualized"],
            len(train_allocations),
            str(bool(metrics["all_cash"])).lower(),
            _format_duration(time.perf_counter() - started_at),
        )
        return PortfolioTrialResult(
            trial_index=task.trial_index,
            state="COMPLETE",
            params=params,
            objective_score=score,
            metrics=metrics,
            daily_row_count=len(train_daily),
            allocation_row_count=len(train_allocations),
            elapsed_seconds=time.perf_counter() - started_at,
        )
    except Exception as exc:
        LOGGER.exception("Portfolio grid trial %d failed", task.trial_index)
        return PortfolioTrialResult(
            trial_index=task.trial_index,
            state="FAILED",
            params=params,
            objective_score=float("-inf"),
            metrics={
                "sharpe_ratio": float("nan"),
                "alpha_over_benchmark_net": float("nan"),
                "calmar_ratio": float("nan"),
                "max_drawdown": float("nan"),
                "turnover_annualized": float("nan"),
                "all_cash": float("nan"),
            },
            daily_row_count=0,
            allocation_row_count=0,
            elapsed_seconds=time.perf_counter() - started_at,
            error_message=str(exc),
        )


def _trial_results_to_ledger(results: list[PortfolioTrialResult], *, lambda_cost: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trial in results:
        row: dict[str, object] = {
            "trial_index": int(trial.trial_index),
            "state": trial.state,
            "objective_score": float(trial.objective_score),
            "lambda_cost": float(lambda_cost),
            "daily_row_count": float(trial.daily_row_count),
            "allocation_row_count": float(trial.allocation_row_count),
            "elapsed_seconds": float(trial.elapsed_seconds),
            "error_message": trial.error_message or "",
        }
        for param_name, param_value in trial.params.items():
            row[param_name] = param_value
        for attr_name, attr_value in trial.metrics.items():
            row[attr_name] = attr_value
        rows.append(row)
    return pd.DataFrame(rows).sort_values("trial_index").reset_index(drop=True)


def _execute_trial_grid(
    trial_grid: list[PortfolioTrialTask],
    *,
    predictions: pd.DataFrame,
    runtime: PortfolioOptimizationConfig,
    config: BacktestConfig,
    covariance_by_lookback: dict[int, pd.DataFrame],
    trial_workers: int,
) -> list[PortfolioTrialResult]:
    results: list[PortfolioTrialResult] = []
    total_trial_count = len(trial_grid)
    study_started_at = time.perf_counter()
    best_score = float("-inf")
    process_context: BaseContext = _resolve_trial_process_context()
    LOGGER.info(
        "Portfolio grid backend=process | workers=%d | start_method=%s",
        trial_workers,
        process_context.get_start_method(),
    )
    with ProcessPoolExecutor(
        max_workers=trial_workers,
        mp_context=process_context,
        initializer=_init_trial_worker,
        initargs=(predictions, runtime, config, covariance_by_lookback),
    ) as executor:
        futures: dict[Future[PortfolioTrialResult], PortfolioTrialTask] = {
            executor.submit(_run_grid_trial_worker, task): task
            for task in trial_grid
        }
        for completed_count, future in enumerate(as_completed(list(futures.keys())), start=1):
            result = future.result()
            results.append(result)
            best_score = max(best_score, float(result.objective_score))
            elapsed_seconds = time.perf_counter() - study_started_at
            average_seconds = elapsed_seconds / max(1, completed_count)
            eta_seconds = average_seconds * max(0, total_trial_count - completed_count)
            LOGGER.info(
                "Portfolio grid progress: %d/%d | last_trial=%d | best_score=%.6f | elapsed=%s | eta=%s",
                completed_count,
                total_trial_count,
                result.trial_index,
                best_score,
                _format_duration(elapsed_seconds),
                _format_duration(eta_seconds),
            )
    return results


def _best_trial_index(trial_ledger: pd.DataFrame) -> int:
    completed = pd.DataFrame(trial_ledger.loc[trial_ledger["state"] == "COMPLETE"].copy())
    source = completed if not completed.empty else trial_ledger
    valid = pd.DataFrame(
        source.loc[
            (source["alpha_over_benchmark_net"] > 0.0)
            & (source["max_drawdown"] >= -0.15)
            & (source["turnover_annualized"] <= 25.0)
        ].copy(),
    )
    ranked_source = valid if not valid.empty else source
    ranked = ranked_source.sort_values(
        ["sharpe_ratio", "alpha_over_benchmark_net", "calmar_ratio"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return _series_int(ranked.iloc[0], "trial_index")


def _series_float(row: pd.Series, key: str) -> float:
    return float(np.asarray(row[key], dtype=np.float64).item())


def _series_int(row: pd.Series, key: str) -> int:
    return int(np.asarray(row[key], dtype=np.int64).item())


def _attach_refined_expected_returns(
    predictions: pd.DataFrame,
    *,
    calibrator: FittedAlphaCalibrator,
    refinement_strategy: str = "binary_gate",
    soft_shifted_floor: float = 0.45,
    rank_blend_lambda: float = 0.50,
) -> pd.DataFrame:
    refined = pd.DataFrame(predictions.copy())
    if PRIMARY_PREDICTION_COLUMN not in refined.columns:
        refined[PRIMARY_PREDICTION_COLUMN] = pd.to_numeric(
            refined[PREDICTION_COLUMN],
            errors="coerce",
        ).fillna(0.0)
    expected = calibrator.transform(
        pd.to_numeric(
            refined[PRIMARY_PREDICTION_COLUMN],
            errors="coerce",
        ).to_numpy(dtype=np.float64),
    )
    refined[EXPECTED_RETURN_COLUMN] = expected
    refined = attach_refined_signal_columns(
        refined,
        strategy=refinement_strategy,
        soft_shifted_floor=soft_shifted_floor,
        rank_blend_lambda=rank_blend_lambda,
    )
    refined[EXPECTED_RETURN_COLUMN] = refined[REFINED_EXPECTED_RETURN_COLUMN]
    return refined


def _load_meta_prediction_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not META_TRAIN_OOF_PREDICTIONS_PARQUET.exists() or not META_VAL_PREDICTIONS_PARQUET.exists():
        raise FileNotFoundError(
            "portfolio_optimization requires meta_labeling artifacts. Missing meta_train_oof_predictions.parquet or meta_val_predictions.parquet.",
        )
    train_predictions = pd.read_parquet(META_TRAIN_OOF_PREDICTIONS_PARQUET)
    val_predictions = pd.read_parquet(META_VAL_PREDICTIONS_PARQUET)
    train_predictions = pd.DataFrame(train_predictions).sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
    val_predictions = pd.DataFrame(val_predictions).sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
    assert_train_only_fit_frame(train_predictions, context="portfolio_optimization.meta_train_predictions")
    if set(val_predictions[SPLIT_COLUMN].astype(str).unique()) != {VAL_SPLIT_NAME}:
        raise ValueError("meta_val_predictions must contain only validation rows.")
    return train_predictions, val_predictions


def run_portfolio_optimization_pipeline(
    config: PortfolioOptimizationConfig | None = None,
    *,
    dataset_path: Path = FEATURE_SELECTION_FILTERED_DATASET_PARQUET,
) -> dict[str, object]:
    del dataset_path
    runtime = config or PortfolioOptimizationConfig()
    base_backtest_config = BacktestConfig()
    train, val = _load_meta_prediction_panels()
    LOGGER.info(
        "Portfolio optimization meta panels ready: train_rows=%d | val_rows=%d | train_dates=%d | val_dates=%d",
        len(train),
        len(val),
        int(train[DATE_COLUMN].nunique()),
        int(val[DATE_COLUMN].nunique()),
    )
    oof_train_predictions = pd.DataFrame(train.copy())
    LOGGER.info(
        "Portfolio alpha input prepared: oof_rows=%d | oof_dates=%d",
        len(oof_train_predictions),
        int(oof_train_predictions[DATE_COLUMN].nunique()),
    )
    alpha_fit_frame = pd.DataFrame(oof_train_predictions.copy())
    alpha_fit_frame[PREDICTION_COLUMN] = pd.to_numeric(
        alpha_fit_frame[PRIMARY_PREDICTION_COLUMN],
        errors="coerce",
    ).fillna(0.0)
    calibrator, alpha_audit = fit_alpha_calibrator_train_only(alpha_fit_frame)
    LOGGER.info("Portfolio alpha calibrator fit: audit_rows=%d", len(alpha_audit))
    covariance_by_lookback = _fit_risk_model_cache(
        train,
        runtime=runtime,
    )
    val_predictions = pd.DataFrame(val.copy())
    LOGGER.info("Portfolio validation panels loaded: rows=%d", len(val_predictions))
    oof_train_predictions = _attach_refined_expected_returns(
        oof_train_predictions,
        calibrator=calibrator,
    )
    val_predictions = _attach_refined_expected_returns(
        val_predictions,
        calibrator=calibrator,
    )
    best_params: dict[str, object] = {}
    if oof_train_predictions[DATE_COLUMN].nunique() < runtime.oof_min_train_dates:
        raise ValueError("portfolio_optimization requires at least oof_min_train_dates mature train dates.")
    trial_grid = _build_trial_grid(runtime)
    total_trial_count = len(trial_grid)
    trial_workers = resolve_executor_worker_count(
        task_count=total_trial_count,
        requested_workers=runtime.trial_parallel_workers,
    )
    LOGGER.info(
        "Portfolio grid search started: trials=%d | workers=%d | objective=penalized_sharpe",
        total_trial_count,
        trial_workers,
    )
    study_started_at = time.perf_counter()
    trial_results = _execute_trial_grid(
        trial_grid,
        predictions=oof_train_predictions,
        runtime=runtime,
        config=base_backtest_config,
        covariance_by_lookback=covariance_by_lookback,
        trial_workers=trial_workers,
    )
    trial_ledger = _trial_results_to_ledger(trial_results, lambda_cost=runtime.lambda_cost)
    selected_trial_index = _best_trial_index(trial_ledger)
    selected_row = pd.DataFrame(trial_ledger.loc[trial_ledger["trial_index"] == selected_trial_index]).iloc[0]
    selected_covariance_lookback_days = _series_int(selected_row, "covariance_lookback_days")
    selected_lambda_risk = _series_float(selected_row, "lambda_risk")
    selected_lambda_turnover = _series_float(selected_row, "lambda_turnover")
    selected_lambda_cost = _series_float(selected_row, "lambda_cost")
    selected_max_position_weight = _series_float(selected_row, "max_position_weight")
    selected_max_sector_weight = _series_float(selected_row, "max_sector_weight")
    selected_no_trade_buffer_bps = _series_float(selected_row, "no_trade_buffer_bps")
    best_params = {
        "trial_index": int(selected_trial_index),
        "portfolio_construction_mode": "optimizer_miqp",
        "solver_backend": "scip_miqp",
        "alpha_calibration_method": "isotonic",
        "alpha_calibrator": serialize_alpha_calibrator(calibrator),
        "meta_probability_column": META_PROBABILITY_COLUMN,
        "lambda_risk": selected_lambda_risk,
        "lambda_turnover": selected_lambda_turnover,
        "lambda_cost": selected_lambda_cost,
        "max_position_weight": selected_max_position_weight,
        "max_sector_weight": selected_max_sector_weight,
        "no_trade_buffer_bps": selected_no_trade_buffer_bps,
        "min_target_weight": float(runtime.min_target_weight),
        "covariance_lookback_days": selected_covariance_lookback_days,
        "covariance_min_history_days": int(runtime.covariance_min_history_days),
        "miqp_time_limit_seconds": float(runtime.miqp_time_limit_seconds),
        "miqp_relative_gap": float(runtime.miqp_relative_gap),
        "miqp_candidate_pool_size": int(runtime.miqp_candidate_pool_size),
        "miqp_primary_objective_tolerance_bps": float(runtime.miqp_primary_objective_tolerance_bps),
    }
    _, best_train_daily, _, best_train_allocations, _ = _score_trial(
        oof_train_predictions,
        config=base_backtest_config,
        runtime=runtime,
        risk_covariance=covariance_by_lookback[selected_covariance_lookback_days],
        lambda_risk=selected_lambda_risk,
        lambda_turnover=selected_lambda_turnover,
        lambda_cost=selected_lambda_cost,
        max_position_weight=selected_max_position_weight,
        max_sector_weight=selected_max_sector_weight,
        no_trade_buffer_bps=selected_no_trade_buffer_bps,
    )
    _, validation_daily, validation_summary, validation_allocations, _ = _score_trial(
        val_predictions,
        config=base_backtest_config,
        runtime=runtime,
        risk_covariance=covariance_by_lookback[selected_covariance_lookback_days],
        lambda_risk=selected_lambda_risk,
        lambda_turnover=selected_lambda_turnover,
        lambda_cost=selected_lambda_cost,
        max_position_weight=selected_max_position_weight,
        max_sector_weight=selected_max_sector_weight,
        no_trade_buffer_bps=selected_no_trade_buffer_bps,
    )
    save_portfolio_optimization_outputs(
        best_params=best_params,
        trial_ledger=trial_ledger,
        train_cv_daily=best_train_daily,
        train_cv_allocations=best_train_allocations,
        validation_daily=validation_daily,
        validation_allocations=validation_allocations,
        validation_summary=validation_summary,
        alpha_calibration_audit=alpha_audit,
        covariance_frame=covariance_by_lookback[selected_covariance_lookback_days],
    )
    LOGGER.info(
        "Portfolio optimization completed in %s | trial=%d | sharpe=%.6f | alpha=%.6f",
        _format_duration(time.perf_counter() - study_started_at),
        selected_trial_index,
        float(validation_summary.get("sharpe_ratio", 0.0)),
        float(validation_summary.get("alpha_over_benchmark_net", 0.0)),
    )
    return {
        "best_params": best_params,
        "validation_summary": validation_summary,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_portfolio_optimization_pipeline()


if __name__ == "__main__":
    main()
