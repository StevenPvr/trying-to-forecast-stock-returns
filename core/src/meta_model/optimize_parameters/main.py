from __future__ import annotations

import gc
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.paths import (
    PREPROCESSED_OUTPUT_PARQUET,
    XGBOOST_OPTUNA_BEST_PARAMS_JSON,
    XGBOOST_OPTUNA_TRIALS_CSV,
    XGBOOST_OPTUNA_TRIALS_PARQUET,
)
from core.src.meta_model.optimize_parameters.config import OptimizationConfig, OPTUNA_STUDY_NAME
from core.src.meta_model.optimize_parameters.cv import WalkForwardFold, build_walk_forward_folds
from core.src.meta_model.optimize_parameters.dataset import (
    OptimizationDatasetBundle,
    build_optimization_dataset_bundle,
    load_preprocessed_dataset,
)
from core.src.meta_model.optimize_parameters.io import save_optimization_outputs
from core.src.meta_model.optimize_parameters.parallelism import ParallelismPlan, resolve_parallelism
from core.src.meta_model.optimize_parameters.robustness import (
    aggregate_robust_objective,
    build_train_windows,
    compute_complexity_penalty,
)
from core.src.meta_model.optimize_parameters.search_space import (
    load_optuna_module,
    load_xgboost_module,
    suggest_xgboost_params,
)
from core.src.meta_model.optimize_parameters.selection import select_one_standard_error_trial

LOGGER: logging.Logger = logging.getLogger(__name__)


def _format_duration(seconds: float) -> str:
    rounded_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(rounded_seconds, 60)
    hours, mins = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{mins:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _evaluate_single_fold(
    dataset_bundle: OptimizationDatasetBundle,
    fold: WalkForwardFold,
    booster_params: dict[str, Any],
    optimization_config: OptimizationConfig,
) -> dict[str, Any]:
    xgb = load_xgboost_module()
    feature_columns = dataset_bundle.feature_columns
    validation_matrix = xgb.DMatrix(
        np.ascontiguousarray(dataset_bundle.feature_matrix[fold.validation_indices]),
        label=np.ascontiguousarray(dataset_bundle.target_array[fold.validation_indices]),
        feature_names=feature_columns,
    )
    train_windows = build_train_windows(
        dataset_bundle.metadata,
        fold.train_indices,
        random_seed=optimization_config.random_seed + fold.index,
        recent_tail_fraction=optimization_config.recent_train_tail_fraction,
        random_window_count=optimization_config.random_train_window_count,
        random_window_min_fraction=optimization_config.random_train_window_min_fraction,
    )
    window_rows: list[dict[str, Any]] = []
    for train_window in train_windows:
        train_matrix = xgb.DMatrix(
            np.ascontiguousarray(dataset_bundle.feature_matrix[train_window.train_indices]),
            label=np.ascontiguousarray(dataset_bundle.target_array[train_window.train_indices]),
            feature_names=feature_columns,
        )
        evaluation_history: dict[str, dict[str, list[float]]] = {}
        booster = xgb.train(
            params=booster_params,
            dtrain=train_matrix,
            num_boost_round=optimization_config.boost_rounds,
            evals=[(validation_matrix, "validation")],
            evals_result=evaluation_history,
            callbacks=[
                xgb.callback.EarlyStopping(
                    rounds=optimization_config.early_stopping_rounds,
                    metric_name="rmse",
                    data_name="validation",
                    save_best=True,
                ),
            ],
            verbose_eval=False,
        )
        window_rows.append({
            "label": train_window.label,
            "rmse": float(booster.best_score),
            "best_iteration": int(booster.best_iteration),
            "coverage_fraction": train_window.coverage_fraction,
        })
        del train_matrix
        del booster
        del evaluation_history

    del validation_matrix
    gc.collect()

    window_rmse = np.asarray([row["rmse"] for row in window_rows], dtype=np.float64)
    window_best_iterations = np.asarray(
        [row["best_iteration"] for row in window_rows],
        dtype=np.float64,
    )
    full_window = next(row for row in window_rows if row["label"] == "full")
    return {
        "fold_index": fold.index,
        "fold_weight": fold.weight,
        "rmse": float(window_rmse.mean()),
        "rmse_full_window": float(full_window["rmse"]),
        "rmse_window_std": float(window_rmse.std(ddof=0)),
        "best_iteration": int(round(float(window_best_iterations.mean()))),
        "best_iteration_full_window": int(full_window["best_iteration"]),
        "window_count": len(window_rows),
        "train_row_count": fold.train_row_count,
        "validation_row_count": fold.validation_row_count,
        "validation_start_date": fold.validation_start_date.isoformat(),
        "validation_end_date": fold.validation_end_date.isoformat(),
    }


def _evaluate_trial_folds(
    dataset_bundle: OptimizationDatasetBundle,
    folds: list[WalkForwardFold],
    booster_params: dict[str, Any],
    optimization_config: OptimizationConfig,
    parallelism_plan: ParallelismPlan,
) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=parallelism_plan.fold_workers) as executor:
        futures = [
            executor.submit(
                _evaluate_single_fold,
                dataset_bundle,
                fold,
                booster_params,
                optimization_config,
            )
            for fold in folds
        ]
        results = [future.result() for future in futures]
    return sorted(results, key=lambda row: int(row["fold_index"]))


def _build_optuna_objective(
    dataset_bundle: OptimizationDatasetBundle,
    folds: list[WalkForwardFold],
    optimization_config: OptimizationConfig,
    parallelism_plan: ParallelismPlan,
):
    def objective(trial: Any) -> float:
        booster_params = suggest_xgboost_params(
            trial,
            threads_per_fold=parallelism_plan.threads_per_fold,
            random_seed=optimization_config.random_seed,
        )
        started_at = time.perf_counter()
        fold_results = _evaluate_trial_folds(
            dataset_bundle,
            folds,
            booster_params,
            optimization_config,
            parallelism_plan,
        )
        complexity_penalty = compute_complexity_penalty(
            max_depth=int(booster_params["max_depth"]),
            average_best_iteration=float(
                np.mean([float(result["best_iteration"]) for result in fold_results]),
            ),
            boost_rounds=optimization_config.boost_rounds,
            penalty_alpha=optimization_config.complexity_penalty_alpha,
        )
        aggregate = aggregate_robust_objective(
            fold_rmse=[float(result["rmse"]) for result in fold_results],
            fold_weights=[float(result["fold_weight"]) for result in fold_results],
            fold_window_std=[float(result["rmse_window_std"]) for result in fold_results],
            stability_penalty_alpha=optimization_config.stability_penalty_alpha,
            train_window_stability_alpha=optimization_config.train_window_stability_alpha,
            complexity_penalty=complexity_penalty,
            bootstrap_samples=optimization_config.objective_standard_error_bootstrap_samples,
            random_seed=optimization_config.random_seed,
        )
        for result in fold_results:
            fold_index = int(result["fold_index"])
            trial.set_user_attr(f"fold_{fold_index}_rmse", float(result["rmse"]))
            trial.set_user_attr(f"fold_{fold_index}_rmse_full_window", float(result["rmse_full_window"]))
            trial.set_user_attr(f"fold_{fold_index}_rmse_window_std", float(result["rmse_window_std"]))
            trial.set_user_attr(f"fold_{fold_index}_best_iteration", int(result["best_iteration"]))
            trial.set_user_attr(f"fold_{fold_index}_window_count", int(result["window_count"]))
        trial.set_user_attr("mean_rmse", aggregate["mean_rmse"])
        trial.set_user_attr("std_rmse", aggregate["std_rmse"])
        trial.set_user_attr("window_std_mean", aggregate["window_std_mean"])
        trial.set_user_attr("objective_standard_error", aggregate["objective_standard_error"])
        trial.set_user_attr("complexity_penalty", complexity_penalty)
        trial.set_user_attr("objective_score", aggregate["objective_score"])
        trial.set_user_attr("elapsed_seconds", time.perf_counter() - started_at)
        LOGGER.info(
            "Optuna trial %s completed | objective=%.6f | mean_rmse=%.6f | std_rmse=%.6f | window_std_mean=%.6f | complexity=%.6f | elapsed=%s",
            trial.number,
            aggregate["objective_score"],
            aggregate["mean_rmse"],
            aggregate["std_rmse"],
            aggregate["window_std_mean"],
            complexity_penalty,
            _format_duration(time.perf_counter() - started_at),
        )
        return aggregate["objective_score"]

    return objective


def _study_to_frame(study: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "trial_number": int(trial.number),
            "state": str(trial.state),
            "objective_score": float(trial.value) if trial.value is not None else np.nan,
        }
        for param_name, param_value in trial.params.items():
            row[f"param_{param_name}"] = param_value
        for attr_name, attr_value in trial.user_attrs.items():
            row[attr_name] = attr_value
        rows.append(row)
    return pd.DataFrame(rows).sort_values("trial_number").reset_index(drop=True)


def optimize_xgboost_parameters(
    dataset_path: Path = PREPROCESSED_OUTPUT_PARQUET,
    optimization_config: OptimizationConfig | None = None,
    *,
    save_outputs_fn: Callable[..., None] | None = None,
    study_name: str = OPTUNA_STUDY_NAME,
    trials_parquet_path: Path | None = None,
    trials_csv_path: Path | None = None,
    best_params_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = optimization_config or OptimizationConfig()
    if save_outputs_fn is None:
        def save_outputs(trials_frame: pd.DataFrame, best_payload: dict[str, Any]) -> None:
            if (
                trials_parquet_path is None
                and trials_csv_path is None
                and best_params_path is None
            ):
                save_optimization_outputs(trials_frame, best_payload)
                return
            save_optimization_outputs(
                trials_frame,
                best_payload,
                trials_parquet_path=trials_parquet_path or XGBOOST_OPTUNA_TRIALS_PARQUET,
                trials_csv_path=trials_csv_path or XGBOOST_OPTUNA_TRIALS_CSV,
                best_params_path=best_params_path or XGBOOST_OPTUNA_BEST_PARAMS_JSON,
            )
    else:
        save_outputs = save_outputs_fn
    started_at = time.perf_counter()
    data = load_preprocessed_dataset(dataset_path)
    dataset_bundle = build_optimization_dataset_bundle(data)
    del data
    feature_columns = dataset_bundle.feature_columns
    folds = build_walk_forward_folds(
        dataset_bundle.metadata,
        fold_count=config.fold_count,
        target_horizon_days=config.target_horizon_days,
    )
    parallelism_plan = resolve_parallelism(
        None,
        config.fold_count,
        dataset_bundle.feature_matrix.nbytes,
    )
    feature_matrix_mb = dataset_bundle.feature_matrix.nbytes / (1024.0 * 1024.0)
    LOGGER.info(
        "Memory-aware parallelism: feature_matrix=%.2f MB | requested_folds=%d | fold_workers=%d | threads_per_fold=%d",
        feature_matrix_mb,
        config.fold_count,
        parallelism_plan.fold_workers,
        parallelism_plan.threads_per_fold,
    )
    LOGGER.info(
        "Starting XGBoost Optuna optimization on %d rows x %d features | folds=%d | fold_workers=%d | threads/fold=%d | trials=%d",
        len(dataset_bundle.metadata),
        len(feature_columns),
        len(folds),
        parallelism_plan.fold_workers,
        parallelism_plan.threads_per_fold,
        config.trial_count,
    )
    for fold in folds:
        LOGGER.info(
            "Walk-forward fold %d/%d | train=%d rows up to %s | val=%d rows from %s to %s | weight=%.1f",
            fold.index,
            len(folds),
            fold.train_row_count,
            fold.train_end_date.date(),
            fold.validation_row_count,
            fold.validation_start_date.date(),
            fold.validation_end_date.date(),
            fold.weight,
        )

    optuna = load_optuna_module()
    sampler = optuna.samplers.TPESampler(seed=config.random_seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
    )
    study.optimize(
        _build_optuna_objective(
            dataset_bundle,
            folds,
            config,
            parallelism_plan,
        ),
        n_trials=config.trial_count,
        n_jobs=1,
        show_progress_bar=False,
    )
    trials_frame = _study_to_frame(study)
    selected_trial = select_one_standard_error_trial(trials_frame)
    best_payload: dict[str, Any] = {
        "objective_score": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "params": study.best_trial.params,
        "user_attrs": study.best_trial.user_attrs,
        "selected_trial_one_standard_error": {
            "trial_number": selected_trial.trial_number,
            "objective_score": selected_trial.objective_score,
            "complexity_penalty": selected_trial.complexity_penalty,
            "params": {
                column.removeprefix("param_"): selected_trial.row[column]
                for column in trials_frame.columns
                if column.startswith("param_")
            },
        },
        "parallelism": asdict(parallelism_plan),
        "config": asdict(config),
        "feature_count": len(feature_columns),
        "study_name": study_name,
    }
    save_outputs(trials_frame, best_payload)
    LOGGER.info(
        "XGBoost Optuna optimization completed in %s | best_trial=%d | selected_one_se_trial=%d | objective=%.6f",
        _format_duration(time.perf_counter() - started_at),
        int(study.best_trial.number),
        selected_trial.trial_number,
        float(study.best_value),
    )
    return trials_frame, best_payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    optimize_xgboost_parameters()


if __name__ == "__main__":
    main()
