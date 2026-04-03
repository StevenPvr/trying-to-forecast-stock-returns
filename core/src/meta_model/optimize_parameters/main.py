from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, TypedDict, cast

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.paths import (
    FEATURE_SELECTION_FILTERED_DATASET_PARQUET,
    OPTIMIZATION_OVERFITTING_REPORT_JSON,
    OPTIMIZATION_TRIAL_LEDGER_CSV,
    OPTIMIZATION_TRIAL_LEDGER_PARQUET,
    XGBOOST_OPTUNA_BEST_PARAMS_JSON,
    XGBOOST_OPTUNA_TRIALS_CSV,
    XGBOOST_OPTUNA_TRIALS_PARQUET,
)
from core.src.meta_model.overfitting import estimate_probability_of_backtest_overfitting
from core.src.meta_model.optimize_parameters.config import OptimizationConfig, OPTUNA_STUDY_NAME
from core.src.meta_model.optimize_parameters.cv import WalkForwardFold, build_walk_forward_folds
from core.src.meta_model.optimize_parameters.dataset import (
    OptimizationDatasetBundle,
    build_optimization_dataset_bundle,
    load_preprocessed_dataset,
)
from core.src.meta_model.optimize_parameters.io import save_optimization_outputs
from core.src.meta_model.optimize_parameters.objective import (
    aggregate_fold_rank_ic,
    bootstrap_rank_ic_objective_standard_error,
)
from core.src.meta_model.optimize_parameters.parallelism import ParallelismPlan, resolve_parallelism
from core.src.meta_model.optimize_parameters.robustness import (
    build_train_windows,
    compute_complexity_penalty,
)
from core.src.meta_model.optimize_parameters.search_space import (
    load_optuna_module,
    load_xgboost_module,
    suggest_xgboost_params,
)
from core.src.meta_model.optimize_parameters.selection import select_one_standard_error_trial
from core.src.meta_model.research_metrics import compute_mean_daily_spearman_ic

LOGGER: logging.Logger = logging.getLogger(__name__)
_process_fold_dataset_bundle: OptimizationDatasetBundle | None = None
_process_fold_booster_params: dict[str, Any] | None = None
_process_fold_optimization_config: OptimizationConfig | None = None


class OverfittingReportPayload(TypedDict):
    trial_count: int
    pbo: float
    selected_trial_number: int
    selected_trial_objective_score: float


SaveOutputsFn = Callable[[pd.DataFrame, dict[str, Any], OverfittingReportPayload], None]


def _format_duration(seconds: float) -> str:
    rounded_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(rounded_seconds, 60)
    hours, mins = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{mins:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _process_pool_available() -> bool:
    return "fork" in mp.get_all_start_methods()


def _install_process_fold_context(
    dataset_bundle: OptimizationDatasetBundle,
    booster_params: dict[str, Any],
    optimization_config: OptimizationConfig,
) -> None:
    global _process_fold_dataset_bundle, _process_fold_booster_params, _process_fold_optimization_config
    _process_fold_dataset_bundle = dataset_bundle
    _process_fold_booster_params = booster_params
    _process_fold_optimization_config = optimization_config


def _clear_process_fold_context() -> None:
    global _process_fold_dataset_bundle, _process_fold_booster_params, _process_fold_optimization_config
    _process_fold_dataset_bundle = None
    _process_fold_booster_params = None
    _process_fold_optimization_config = None


def _evaluate_single_fold_in_process(fold: WalkForwardFold) -> dict[str, Any]:
    if (
        _process_fold_dataset_bundle is None
        or _process_fold_booster_params is None
        or _process_fold_optimization_config is None
    ):
        raise RuntimeError("Optimize-parameters process fold context is not installed.")
    return _evaluate_single_fold(
        _process_fold_dataset_bundle,
        fold,
        _process_fold_booster_params,
        _process_fold_optimization_config,
    )


def _evaluate_single_fold(
    dataset_bundle: OptimizationDatasetBundle,
    fold: WalkForwardFold,
    booster_params: dict[str, Any],
    optimization_config: OptimizationConfig,
) -> dict[str, Any]:
    xgb = load_xgboost_module()
    feature_columns = dataset_bundle.feature_columns
    validation_dates = np.asarray(
        pd.to_datetime(
            cast(pd.Series, dataset_bundle.metadata.iloc[fold.validation_indices]["date"]),
        ),
    )
    validation_matrix = xgb.DMatrix(
        np.ascontiguousarray(dataset_bundle.feature_matrix[fold.validation_indices]),
        label=np.ascontiguousarray(dataset_bundle.target_array[fold.validation_indices]),
        feature_names=feature_columns,
    )

    def _daily_rank_ic_metric(predictions: np.ndarray, dmatrix: Any) -> tuple[str, float]:
        labels = np.asarray(dmatrix.get_label(), dtype=np.float64)
        return (
            "daily_rank_ic",
            compute_mean_daily_spearman_ic(
                np.asarray(predictions, dtype=np.float64),
                labels,
                validation_dates,
            ),
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
            custom_metric=_daily_rank_ic_metric,
            maximize=True,
            evals_result=evaluation_history,
            callbacks=[
                xgb.callback.EarlyStopping(
                    rounds=optimization_config.early_stopping_rounds,
                    metric_name="daily_rank_ic",
                    data_name="validation",
                    maximize=True,
                    save_best=True,
                ),
            ],
            verbose_eval=False,
        )
        window_rows.append({
            "label": train_window.label,
            "daily_rank_ic": float(booster.best_score),
            "best_iteration": int(booster.best_iteration),
            "coverage_fraction": train_window.coverage_fraction,
        })
        del train_matrix
        del booster
        del evaluation_history

    del validation_matrix
    gc.collect()

    window_rank_ic = np.asarray([row["daily_rank_ic"] for row in window_rows], dtype=np.float64)
    window_best_iterations = np.asarray(
        [row["best_iteration"] for row in window_rows],
        dtype=np.float64,
    )
    full_window = next(row for row in window_rows if row["label"] == "full")
    return {
        "fold_index": fold.index,
        "fold_weight": fold.weight,
        "daily_rank_ic": float(window_rank_ic.mean()),
        "daily_rank_ic_full_window": float(full_window["daily_rank_ic"]),
        "daily_rank_ic_window_std": float(window_rank_ic.std(ddof=0)),
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
    if parallelism_plan.fold_workers <= 1 or len(folds) <= 1:
        results = [
            _evaluate_single_fold(
                dataset_bundle,
                fold,
                booster_params,
                optimization_config,
            )
            for fold in folds
        ]
        return sorted(results, key=lambda row: int(row["fold_index"]))
    if _process_pool_available():
        LOGGER.info(
            "Fold evaluation backend=process | workers=%d | threads_per_fold=%d",
            parallelism_plan.fold_workers,
            parallelism_plan.threads_per_fold,
        )
        _install_process_fold_context(dataset_bundle, booster_params, optimization_config)
        try:
            return _evaluate_trial_folds_in_process_pool(
                folds,
                parallelism_plan=parallelism_plan,
            )
        except (NotImplementedError, OSError, PermissionError) as error:
            LOGGER.warning(
                "Fold evaluation process backend unavailable (%s); falling back to threads.",
                error,
            )
        finally:
            _clear_process_fold_context()
    LOGGER.info(
        "Fold evaluation backend=thread | workers=%d | threads_per_fold=%d",
        parallelism_plan.fold_workers,
        parallelism_plan.threads_per_fold,
    )
    return _evaluate_trial_folds_in_thread_pool(
        dataset_bundle,
        folds,
        booster_params,
        optimization_config,
        parallelism_plan=parallelism_plan,
    )


def _evaluate_trial_folds_in_process_pool(
    folds: list[WalkForwardFold],
    *,
    parallelism_plan: ParallelismPlan,
) -> list[dict[str, Any]]:
    process_context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=parallelism_plan.fold_workers,
        mp_context=process_context,
    ) as executor:
        futures = [
            executor.submit(_evaluate_single_fold_in_process, fold)
            for fold in folds
        ]
        results = [future.result() for future in futures]
    return sorted(results, key=lambda row: int(row["fold_index"]))


def _evaluate_trial_folds_in_thread_pool(
    dataset_bundle: OptimizationDatasetBundle,
    folds: list[WalkForwardFold],
    booster_params: dict[str, Any],
    optimization_config: OptimizationConfig,
    *,
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
        objective_standard_error = bootstrap_rank_ic_objective_standard_error(
            fold_rank_ic=[float(result["daily_rank_ic"]) for result in fold_results],
            fold_window_std=[float(result["daily_rank_ic_window_std"]) for result in fold_results],
            stability_penalty_alpha=optimization_config.stability_penalty_alpha,
            train_window_stability_alpha=optimization_config.train_window_stability_alpha,
            bootstrap_samples=optimization_config.objective_standard_error_bootstrap_samples,
            random_seed=optimization_config.random_seed,
        )
        aggregate = aggregate_fold_rank_ic(
            fold_rank_ic=[float(result["daily_rank_ic"]) for result in fold_results],
            fold_window_std=[float(result["daily_rank_ic_window_std"]) for result in fold_results],
            stability_penalty_alpha=optimization_config.stability_penalty_alpha,
            train_window_stability_alpha=optimization_config.train_window_stability_alpha,
            complexity_penalty=complexity_penalty,
            objective_standard_error=objective_standard_error,
        )
        for result in fold_results:
            fold_index = int(result["fold_index"])
            trial.set_user_attr(f"fold_{fold_index}_daily_rank_ic", float(result["daily_rank_ic"]))
            trial.set_user_attr(
                f"fold_{fold_index}_daily_rank_ic_full_window",
                float(result["daily_rank_ic_full_window"]),
            )
            trial.set_user_attr(
                f"fold_{fold_index}_daily_rank_ic_window_std",
                float(result["daily_rank_ic_window_std"]),
            )
            trial.set_user_attr(f"fold_{fold_index}_best_iteration", int(result["best_iteration"]))
            trial.set_user_attr(f"fold_{fold_index}_window_count", int(result["window_count"]))
        trial.set_user_attr("mean_daily_rank_ic", aggregate["mean_rank_ic"])
        trial.set_user_attr("std_daily_rank_ic", aggregate["std_rank_ic"])
        trial.set_user_attr("window_std_mean", aggregate["window_std_mean"])
        trial.set_user_attr("objective_standard_error", aggregate["objective_standard_error"])
        trial.set_user_attr("complexity_penalty", complexity_penalty)
        trial.set_user_attr("objective_score", aggregate["objective_score"])
        trial.set_user_attr("elapsed_seconds", time.perf_counter() - started_at)
        LOGGER.info(
            "Optuna trial %s completed | objective=%.6f | mean_daily_rank_ic=%.6f | std_daily_rank_ic=%.6f | window_std_mean=%.6f | complexity=%.6f | elapsed=%s",
            trial.number,
            aggregate["objective_score"],
            aggregate["mean_rank_ic"],
            aggregate["std_rank_ic"],
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


def _build_default_save_outputs_fn(
    *,
    trials_parquet_path: Path | None,
    trials_csv_path: Path | None,
    best_params_path: Path | None,
) -> SaveOutputsFn:
    def save_outputs(
        trials_frame: pd.DataFrame,
        best_payload: dict[str, Any],
        overfitting_report: OverfittingReportPayload,
    ) -> None:
        if (
            trials_parquet_path is None
            and trials_csv_path is None
            and best_params_path is None
        ):
            _save_outputs_with_default_paths(trials_frame, best_payload, overfitting_report)
            return
        _save_outputs_with_custom_paths(
            trials_frame,
            best_payload,
            overfitting_report,
            trials_parquet_path=trials_parquet_path or XGBOOST_OPTUNA_TRIALS_PARQUET,
            trials_csv_path=trials_csv_path or XGBOOST_OPTUNA_TRIALS_CSV,
            best_params_path=best_params_path or XGBOOST_OPTUNA_BEST_PARAMS_JSON,
        )

    return save_outputs


def _save_outputs_with_default_paths(
    trials_frame: pd.DataFrame,
    best_payload: dict[str, Any],
    overfitting_report: OverfittingReportPayload,
) -> None:
    serialized_overfitting_report: dict[str, Any] = dict(overfitting_report)
    try:
        save_optimization_outputs(
            trials_frame,
            best_payload,
            serialized_overfitting_report,
        )
    except TypeError:
        save_optimization_outputs(trials_frame, best_payload)


def _save_outputs_with_custom_paths(
    trials_frame: pd.DataFrame,
    best_payload: dict[str, Any],
    overfitting_report: OverfittingReportPayload,
    *,
    trials_parquet_path: Path,
    trials_csv_path: Path,
    best_params_path: Path,
) -> None:
    serialized_overfitting_report: dict[str, Any] = dict(overfitting_report)
    try:
        save_optimization_outputs(
            trials_frame,
            best_payload,
            serialized_overfitting_report,
            trials_parquet_path=trials_parquet_path,
            trials_csv_path=trials_csv_path,
            best_params_path=best_params_path,
            trial_ledger_parquet_path=OPTIMIZATION_TRIAL_LEDGER_PARQUET,
            trial_ledger_csv_path=OPTIMIZATION_TRIAL_LEDGER_CSV,
            overfitting_report_path=OPTIMIZATION_OVERFITTING_REPORT_JSON,
        )
    except TypeError:
        save_optimization_outputs(
            trials_frame,
            best_payload,
            trials_parquet_path=trials_parquet_path,
            trials_csv_path=trials_csv_path,
            best_params_path=best_params_path,
        )


def _resolve_save_outputs_fn(
    save_outputs_fn: SaveOutputsFn | None,
    *,
    trials_parquet_path: Path | None,
    trials_csv_path: Path | None,
    best_params_path: Path | None,
) -> SaveOutputsFn:
    if save_outputs_fn is not None:
        return save_outputs_fn
    return _build_default_save_outputs_fn(
        trials_parquet_path=trials_parquet_path,
        trials_csv_path=trials_csv_path,
        best_params_path=best_params_path,
    )


def _load_dataset_bundle_with_plan(
    dataset_path: Path,
    config: OptimizationConfig,
) -> tuple[OptimizationDatasetBundle, list[str], list[WalkForwardFold], ParallelismPlan]:
    data = load_preprocessed_dataset(dataset_path)
    dataset_bundle = build_optimization_dataset_bundle(data, dataset_path)
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
    return dataset_bundle, feature_columns, folds, parallelism_plan


def _log_optimization_plan(
    dataset_bundle: OptimizationDatasetBundle,
    feature_columns: list[str],
    folds: list[WalkForwardFold],
    parallelism_plan: ParallelismPlan,
    config: OptimizationConfig,
) -> None:
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


def _run_optuna_study(
    dataset_bundle: OptimizationDatasetBundle,
    folds: list[WalkForwardFold],
    config: OptimizationConfig,
    parallelism_plan: ParallelismPlan,
    study_name: str,
) -> Any:
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
    return study


def _build_best_payload(
    study: Any,
    trials_frame: pd.DataFrame,
    selected_trial: Any,
    parallelism_plan: ParallelismPlan,
    config: OptimizationConfig,
    *,
    feature_count: int,
    study_name: str,
) -> dict[str, Any]:
    return {
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
                for column in cast(list[str], trials_frame.columns.tolist())
                if column.startswith("param_")
            },
        },
        "parallelism": asdict(parallelism_plan),
        "config": asdict(config),
        "feature_count": feature_count,
        "study_name": study_name,
    }


def _build_overfitting_report(
    trials_frame: pd.DataFrame,
    selected_trial: Any,
) -> OverfittingReportPayload:
    return {
        "trial_count": int(len(trials_frame)),
        "pbo": float(estimate_probability_of_backtest_overfitting(trials_frame)),
        "selected_trial_number": int(selected_trial.trial_number),
        "selected_trial_objective_score": float(selected_trial.objective_score),
    }


def optimize_xgboost_parameters(
    dataset_path: Path = FEATURE_SELECTION_FILTERED_DATASET_PARQUET,
    optimization_config: OptimizationConfig | None = None,
    *,
    save_outputs_fn: SaveOutputsFn | None = None,
    study_name: str = OPTUNA_STUDY_NAME,
    trials_parquet_path: Path | None = None,
    trials_csv_path: Path | None = None,
    best_params_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = optimization_config or OptimizationConfig()
    save_outputs = _resolve_save_outputs_fn(
        save_outputs_fn,
        trials_parquet_path=trials_parquet_path,
        trials_csv_path=trials_csv_path,
        best_params_path=best_params_path,
    )
    started_at = time.perf_counter()
    dataset_bundle, feature_columns, folds, parallelism_plan = _load_dataset_bundle_with_plan(
        dataset_path,
        config,
    )
    _log_optimization_plan(
        dataset_bundle,
        feature_columns,
        folds,
        parallelism_plan,
        config,
    )
    study = _run_optuna_study(
        dataset_bundle,
        folds,
        config,
        parallelism_plan,
        study_name,
    )
    trials_frame = _study_to_frame(study)
    selected_trial = select_one_standard_error_trial(trials_frame)
    best_payload = _build_best_payload(
        study,
        trials_frame,
        selected_trial,
        parallelism_plan,
        config,
        feature_count=len(feature_columns),
        study_name=study_name,
    )
    overfitting_report = _build_overfitting_report(trials_frame, selected_trial)
    save_outputs(trials_frame, best_payload, overfitting_report)
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
