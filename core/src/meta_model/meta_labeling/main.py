from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from core.src.meta_model.meta_labeling.metrics import has_binary_label_support
from core.src.meta_model.meta_labeling.study import (
    build_meta_folds,
    mapping_float,
    mapping_int,
    meta_params_to_xgboost,
    run_meta_optuna_study,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.paths import (
    FEATURE_SELECTION_FILTERED_DATASET_PARQUET,
    META_PRIMARY_OOS_TRAIN_TAIL_PARQUET,
    META_PRIMARY_OOS_VAL_PARQUET,
)
from core.src.meta_model.evaluate.dataset import (
    build_feature_columns,
    load_manifest_feature_columns,
    load_preprocessed_evaluation_dataset,
)
from core.src.meta_model.evaluate.training import build_available_training_frame
from core.src.meta_model.meta_labeling.calibration import (
    fit_probability_calibrator_train_only,
    serialize_probability_calibrator,
)
from core.src.meta_model.meta_labeling.config import MetaLabelingConfig
from core.src.meta_model.meta_labeling.features import (
    META_PROBABILITY_COLUMN,
    attach_refined_signal_columns,
    build_meta_feature_columns,
    build_primary_context_columns,
)
from core.src.meta_model.meta_labeling.io import save_meta_labeling_outputs
from core.src.meta_model.meta_labeling.labels import (
    META_LABEL_COLUMN,
    attach_meta_labels,
    select_meta_candidate_rows,
)
from core.src.meta_model.meta_labeling.model import (
    fit_meta_model,
    predict_meta_model,
    serialize_meta_model_artifact,
)
from core.src.meta_model.model_contract import (
    DATE_COLUMN,
    LABEL_EMBARGO_DAYS,
    MODEL_TARGET_COLUMN,
    PREDICTION_COLUMN,
    REALIZED_RETURN_COLUMN,
    SPLIT_COLUMN,
    TEST_SPLIT_NAME,
    TICKER_COLUMN,
    TRAIN_SPLIT_NAME,
    VAL_SPLIT_NAME,
    WEEK_HOLD_EXCESS_RETURN_COLUMN,
    WEEK_HOLD_NET_RETURN_COLUMN,
)
from core.src.meta_model.model_registry.main import ModelSpec, fit_model, predict_model
from core.src.meta_model.split_guard import assert_train_only_fit_frame

LOGGER: logging.Logger = logging.getLogger(__name__)


def _build_required_dataset_columns(
    manifest_feature_columns: list[str],
) -> list[str]:
    columns = [
        DATE_COLUMN,
        TICKER_COLUMN,
        SPLIT_COLUMN,
        MODEL_TARGET_COLUMN,
        REALIZED_RETURN_COLUMN,
        WEEK_HOLD_NET_RETURN_COLUMN,
        WEEK_HOLD_EXCESS_RETURN_COLUMN,
        "stock_close_log_return_lag_1d",
        "company_sector",
        "stock_open_price",
        "hl_context_stock_open_price",
        "stock_trading_volume",
        "hl_context_stock_trading_volume",
        *manifest_feature_columns,
    ]
    return list(dict.fromkeys(columns))


def _load_train_val_dataset(dataset_path: Path) -> tuple[pd.DataFrame, list[str]]:
    manifest_feature_columns = load_manifest_feature_columns(dataset_path)
    required_columns = _build_required_dataset_columns(manifest_feature_columns)
    dataset = load_preprocessed_evaluation_dataset(
        dataset_path,
        allowed_splits=(TRAIN_SPLIT_NAME, VAL_SPLIT_NAME),
        columns=required_columns,
    )
    if TEST_SPLIT_NAME in set(dataset[SPLIT_COLUMN].astype(str).unique()):
        raise ValueError("meta_labeling must not load test rows.")
    feature_columns = manifest_feature_columns or build_feature_columns(dataset)
    ordered = dataset.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
    LOGGER.info(
        "Train/val dataset loaded: rows=%d | features=%d | splits=%s",
        len(ordered),
        len(feature_columns),
        sorted(ordered[SPLIT_COLUMN].astype(str).unique().tolist()),
    )
    return ordered, feature_columns


def _resolve_primary_burn_start_date(
    data: pd.DataFrame,
    *,
    burn_fraction: float,
) -> pd.Timestamp:
    train_dates = pd.Index(
        pd.to_datetime(
            data.loc[data[SPLIT_COLUMN] == TRAIN_SPLIT_NAME, DATE_COLUMN],
        ).drop_duplicates().sort_values(),
    )
    if len(train_dates) == 0:
        raise ValueError("meta_labeling requires train dates.")
    burn_count = max(1, int(math.ceil(len(train_dates) * burn_fraction)))
    if burn_count >= len(train_dates):
        raise ValueError("burn_in_train_fraction leaves no train dates for OOS predictions.")
    burn_start: pd.Timestamp = pd.Timestamp(train_dates.tolist()[burn_count])
    LOGGER.info(
        "Primary burn-in resolved: train_dates=%d | burn_count=%d (%.1f%%) | oos_start=%s",
        len(train_dates),
        burn_count,
        100.0 * burn_fraction,
        burn_start.date(),
    )
    return burn_start


def _build_primary_model_spec() -> ModelSpec:
    from core.src.meta_model.evaluate.parameters import load_selected_xgboost_configuration

    selected = load_selected_xgboost_configuration()
    return ModelSpec(
        model_name="xgboost",
        params=dict(selected.params),
        training_rounds=selected.training_rounds,
    )


def _predict_primary_frame(
    artifact: ModelArtifact,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    predicted = pd.DataFrame(frame.copy())
    predictions = predict_model(artifact, predicted, feature_columns)
    predicted[PREDICTION_COLUMN] = predictions
    return build_primary_context_columns(predicted)


def _iter_primary_prediction_dates(
    data: pd.DataFrame,
    *,
    burn_start_date: pd.Timestamp,
) -> list[pd.Timestamp]:
    unique_dates = pd.Index(pd.to_datetime(data[DATE_COLUMN]).drop_duplicates().sort_values())
    return [pd.Timestamp(value) for value in unique_dates.tolist() if pd.Timestamp(value) >= burn_start_date]


def _resolve_primary_parallel_workers(
    *,
    prediction_dates: list[pd.Timestamp],
    requested_workers: int | None,
) -> int:
    cpu_workers = max(1, os.cpu_count() or 1)
    configured_workers = cpu_workers if requested_workers is None else max(1, requested_workers)
    return max(1, min(len(prediction_dates), configured_workers))


def _build_parallel_primary_model_spec(
    model_spec: ModelSpec,
) -> ModelSpec:
    if model_spec.model_name != "xgboost":
        return model_spec
    parallel_params = dict(model_spec.params)
    parallel_params["nthread"] = 1
    return ModelSpec(
        model_name=model_spec.model_name,
        params=parallel_params,
        target_column=model_spec.target_column,
        training_rounds=model_spec.training_rounds,
    )


def _build_primary_prediction_part(
    *,
    data: pd.DataFrame,
    feature_columns: list[str],
    model_spec: ModelSpec,
    prediction_date: pd.Timestamp,
) -> pd.DataFrame | None:
    try:
        train_frame = build_available_training_frame(
            data,
            prediction_date=prediction_date,
            label_embargo_days=LABEL_EMBARGO_DAYS,
        )
    except ValueError as exc:
        if "Not enough historical dates" in str(exc):
            return None
        raise
    train_frame = pd.DataFrame(train_frame.loc[train_frame[SPLIT_COLUMN] == TRAIN_SPLIT_NAME].copy())
    assert_train_only_fit_frame(train_frame, context="meta_labeling.primary_oos.fit")
    prediction_frame = pd.DataFrame(data.loc[data[DATE_COLUMN] == prediction_date].copy())
    artifact = fit_model(model_spec, train_frame, feature_columns)
    return _predict_primary_frame(artifact, prediction_frame, feature_columns)


def _build_primary_oos_panels(
    data: pd.DataFrame,
    *,
    feature_columns: list[str],
    model_spec: ModelSpec,
    burn_fraction: float,
    parallel_workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    burn_start_date = _resolve_primary_burn_start_date(data, burn_fraction=burn_fraction)
    prediction_dates: list[pd.Timestamp] = _iter_primary_prediction_dates(
        data, burn_start_date=burn_start_date,
    )
    resolved_workers = _resolve_primary_parallel_workers(
        prediction_dates=prediction_dates,
        requested_workers=parallel_workers,
    )
    worker_model_spec = (
        _build_parallel_primary_model_spec(model_spec)
        if resolved_workers > 1
        else model_spec
    )
    LOGGER.info(
        "Building primary OOS panels: prediction_dates=%d | model=%s | workers=%d",
        len(prediction_dates),
        model_spec.model_name,
        resolved_workers,
    )
    parts: list[pd.DataFrame] = []
    started_at: float = time.perf_counter()
    if resolved_workers == 1:
        for date_index, prediction_date in enumerate(prediction_dates, start=1):
            predicted = _build_primary_prediction_part(
                data=data,
                feature_columns=feature_columns,
                model_spec=worker_model_spec,
                prediction_date=prediction_date,
            )
            if predicted is None:
                continue
            parts.append(predicted)
            if date_index == 1 or date_index % 50 == 0 or date_index == len(prediction_dates):
                elapsed: float = time.perf_counter() - started_at
                avg: float = elapsed / date_index
                eta: float = avg * (len(prediction_dates) - date_index)
                LOGGER.info(
                    "Primary OOS retrain: %d/%d dates (%.1f%%) | date=%s | elapsed=%.0fs | eta=%.0fs",
                    date_index,
                    len(prediction_dates),
                    100.0 * date_index / len(prediction_dates),
                    prediction_date.date(),
                    elapsed,
                    eta,
                )
    else:
        LOGGER.info(
            "Primary OOS parallel backend=thread | workers=%d | xgboost_threads_per_worker=%d",
            resolved_workers,
            1 if worker_model_spec.model_name == "xgboost" else max(1, os.cpu_count() or 1),
        )
        future_to_date: dict[Future[pd.DataFrame | None], pd.Timestamp] = {}
        with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
            for prediction_date in prediction_dates:
                future = executor.submit(
                    _build_primary_prediction_part,
                    data=data,
                    feature_columns=feature_columns,
                    model_spec=worker_model_spec,
                    prediction_date=prediction_date,
                )
                future_to_date[future] = prediction_date
            for completed_count, future in enumerate(as_completed(list(future_to_date.keys())), start=1):
                predicted = future.result()
                if predicted is not None:
                    parts.append(predicted)
                prediction_date = future_to_date[future]
                if (
                    completed_count == 1
                    or completed_count % 50 == 0
                    or completed_count == len(prediction_dates)
                ):
                    elapsed = time.perf_counter() - started_at
                    avg = elapsed / completed_count
                    eta = avg * (len(prediction_dates) - completed_count)
                    LOGGER.info(
                        "Primary OOS retrain: %d/%d dates (%.1f%%) | date=%s | elapsed=%.0fs | eta=%.0fs",
                        completed_count,
                        len(prediction_dates),
                        100.0 * completed_count / len(prediction_dates),
                        prediction_date.date(),
                        elapsed,
                        eta,
                    )
    if not parts:
        raise ValueError("meta_labeling could not build any primary OOS predictions.")
    combined = pd.concat(parts, ignore_index=True).sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
    train_tail = pd.DataFrame(combined.loc[combined[SPLIT_COLUMN] == TRAIN_SPLIT_NAME].copy())
    val_panel = pd.DataFrame(combined.loc[combined[SPLIT_COLUMN] == VAL_SPLIT_NAME].copy())
    LOGGER.info(
        "Primary OOS panels built: train_tail=%d rows | val=%d rows | elapsed=%.1fs",
        len(train_tail),
        len(val_panel),
        time.perf_counter() - started_at,
    )
    return train_tail, val_panel


def _load_or_build_primary_oos_panels(
    data: pd.DataFrame,
    *,
    feature_columns: list[str],
    model_spec: ModelSpec,
    burn_fraction: float,
    parallel_workers: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_cached_columns = {
        DATE_COLUMN,
        TICKER_COLUMN,
        SPLIT_COLUMN,
        PREDICTION_COLUMN,
        "primary_prediction",
        "primary_prediction_rank_cs",
        "primary_prediction_zscore_cs",
        "primary_prediction_abs",
        "primary_prediction_sign",
        WEEK_HOLD_NET_RETURN_COLUMN,
    }
    if META_PRIMARY_OOS_TRAIN_TAIL_PARQUET.exists() and META_PRIMARY_OOS_VAL_PARQUET.exists():
        cached_train = pd.read_parquet(META_PRIMARY_OOS_TRAIN_TAIL_PARQUET)
        cached_val = pd.read_parquet(META_PRIMARY_OOS_VAL_PARQUET)
        train_tail = pd.DataFrame(cached_train).sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
        val_panel = pd.DataFrame(cached_val).sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
        missing_cached_columns = sorted(
            required_cached_columns.difference(set(train_tail.columns)).union(
                required_cached_columns.difference(set(val_panel.columns)),
            ),
        )
        if train_tail.empty or val_panel.empty:
            LOGGER.warning(
                "Primary OOS cache exists but is empty; rebuilding panels: train=%s | val=%s",
                META_PRIMARY_OOS_TRAIN_TAIL_PARQUET,
                META_PRIMARY_OOS_VAL_PARQUET,
            )
        elif missing_cached_columns:
            LOGGER.warning(
                "Primary OOS cache missing required columns; rebuilding panels: missing=%s | train=%s | val=%s",
                ", ".join(missing_cached_columns),
                META_PRIMARY_OOS_TRAIN_TAIL_PARQUET,
                META_PRIMARY_OOS_VAL_PARQUET,
            )
        else:
            LOGGER.info(
                "Reusing cached primary OOS panels: train_rows=%d | val_rows=%d | train_path=%s | val_path=%s",
                len(train_tail),
                len(val_panel),
                META_PRIMARY_OOS_TRAIN_TAIL_PARQUET,
                META_PRIMARY_OOS_VAL_PARQUET,
            )
            return train_tail, val_panel
    return _build_primary_oos_panels(
        data,
        feature_columns=feature_columns,
        model_spec=model_spec,
        burn_fraction=burn_fraction,
        parallel_workers=parallel_workers,
    )


def _build_meta_training_frame(
    panel: pd.DataFrame,
    *,
    runtime: MetaLabelingConfig,
) -> pd.DataFrame:
    return attach_meta_labels(
        panel,
        primary_prediction_threshold=runtime.meta_primary_candidate_threshold,
        minimum_target_net_return=runtime.meta_min_target_net_return,
    )


def _build_meta_oof_predictions(
    train_tail: pd.DataFrame,
    *,
    meta_feature_columns: list[str],
    runtime: MetaLabelingConfig,
    best_params: dict[str, object],
) -> pd.DataFrame:
    params = meta_params_to_xgboost(
        mapping_float(best_params, "learning_rate"),
        mapping_int(best_params, "max_depth"),
        mapping_float(best_params, "min_child_weight"),
        random_seed=runtime.random_seed,
        subsample=mapping_float(best_params, "subsample"),
        colsample_bytree=mapping_float(best_params, "colsample_bytree"),
        gamma=mapping_float(best_params, "gamma"),
        reg_lambda=mapping_float(best_params, "lambda"),
        reg_alpha=mapping_float(best_params, "alpha"),
        max_bin=mapping_int(best_params, "max_bin"),
        scale_pos_weight=mapping_float(best_params, "scale_pos_weight"),
    )
    selected_training_rounds = mapping_int(best_params, "selected_training_rounds")
    parts: list[pd.DataFrame] = []
    for train_cutoff, fold_dates in build_meta_folds(train_tail, fold_count=runtime.fold_count):
        fold_train = pd.DataFrame(train_tail.loc[train_tail[DATE_COLUMN] <= train_cutoff].copy())
        fold_val = pd.DataFrame(train_tail.loc[train_tail[DATE_COLUMN].isin(fold_dates)].copy())
        fold_train_candidates = select_meta_candidate_rows(fold_train)
        fold_val_candidates = select_meta_candidate_rows(fold_val)
        scored = pd.DataFrame(fold_val.copy())
        scored[META_PROBABILITY_COLUMN] = 0.0
        if (
            fold_train_candidates.empty
            or fold_val_candidates.empty
            or not has_binary_label_support(fold_train_candidates)
        ):
            parts.append(scored)
            continue
        artifact = fit_meta_model(
            fold_train_candidates,
            meta_feature_columns,
            label_column=META_LABEL_COLUMN,
            params=params,
            training_rounds=selected_training_rounds,
            early_stopping_rounds=None,
            early_stopping_validation_fraction=runtime.early_stopping_validation_fraction,
            minimum_training_rounds=runtime.minimum_training_rounds,
        )
        scored.loc[fold_val_candidates.index, META_PROBABILITY_COLUMN] = predict_meta_model(
            artifact,
            fold_val_candidates,
        )
        parts.append(scored)
    if not parts:
        raise ValueError("meta_labeling could not produce OOF meta predictions.")
    oof_predictions: pd.DataFrame = pd.concat(parts, ignore_index=True).sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
    LOGGER.info(
        "Meta OOF predictions built: rows=%d | folds=%d",
        len(oof_predictions),
        len(parts),
    )
    return oof_predictions


def _fit_final_meta_outputs(
    train_tail: pd.DataFrame,
    val_panel: pd.DataFrame,
    *,
    meta_feature_columns: list[str],
    runtime: MetaLabelingConfig,
    best_params: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame]:
    params = meta_params_to_xgboost(
        mapping_float(best_params, "learning_rate"),
        mapping_int(best_params, "max_depth"),
        mapping_float(best_params, "min_child_weight"),
        random_seed=runtime.random_seed,
        subsample=mapping_float(best_params, "subsample"),
        colsample_bytree=mapping_float(best_params, "colsample_bytree"),
        gamma=mapping_float(best_params, "gamma"),
        reg_lambda=mapping_float(best_params, "lambda"),
        reg_alpha=mapping_float(best_params, "alpha"),
        max_bin=mapping_int(best_params, "max_bin"),
        scale_pos_weight=mapping_float(best_params, "scale_pos_weight"),
    )
    selected_training_rounds = mapping_int(best_params, "selected_training_rounds")
    train_tail_candidates = select_meta_candidate_rows(train_tail)
    artifact = fit_meta_model(
        train_tail_candidates,
        meta_feature_columns,
        label_column=META_LABEL_COLUMN,
        params=params,
        training_rounds=selected_training_rounds,
        early_stopping_rounds=None,
        early_stopping_validation_fraction=runtime.early_stopping_validation_fraction,
        minimum_training_rounds=runtime.minimum_training_rounds,
    )
    scored_val = pd.DataFrame(val_panel.copy())
    scored_val[META_PROBABILITY_COLUMN] = 0.0
    val_candidates = select_meta_candidate_rows(val_panel)
    if not val_candidates.empty:
        scored_val.loc[val_candidates.index, META_PROBABILITY_COLUMN] = predict_meta_model(
            artifact,
            val_candidates,
        )
    LOGGER.info(
        "Final meta model fitted on train_tail=%d rows | val scored=%d rows",
        len(train_tail),
        len(scored_val),
    )
    return serialize_meta_model_artifact(artifact), scored_val


def run_meta_labeling_pipeline(
    config: MetaLabelingConfig | None = None,
    *,
    dataset_path: Path = FEATURE_SELECTION_FILTERED_DATASET_PARQUET,
) -> dict[str, object]:
    pipeline_started_at: float = time.perf_counter()
    runtime = config or MetaLabelingConfig()
    LOGGER.info("Meta-labeling pipeline started: dataset_path=%s", dataset_path)
    dataset, feature_columns = _load_train_val_dataset(dataset_path)
    model_spec = _build_primary_model_spec()
    LOGGER.info(
        "Meta-labeling dataset ready: rows=%d | features=%d | model=%s",
        len(dataset),
        len(feature_columns),
        model_spec.model_name,
    )
    primary_train_tail, primary_val = _load_or_build_primary_oos_panels(
        dataset,
        feature_columns=feature_columns,
        model_spec=model_spec,
        burn_fraction=runtime.burn_in_train_fraction,
        parallel_workers=runtime.primary_oos_parallel_workers,
    )
    meta_feature_columns = build_meta_feature_columns(feature_columns)
    meta_train_panel = _build_meta_training_frame(
        primary_train_tail,
        runtime=runtime,
    )
    assert_train_only_fit_frame(meta_train_panel, context="meta_labeling.train_panel")
    meta_val_panel = _build_meta_training_frame(
        primary_val,
        runtime=runtime,
    )
    unique_train_dates = pd.Index(
        pd.to_datetime(meta_train_panel[DATE_COLUMN]).drop_duplicates().sort_values(),
    )
    calibration_date_count = max(
        1,
        int(math.ceil(len(unique_train_dates) * runtime.calibration_holdout_fraction)),
    )
    calibration_dates = set(unique_train_dates[-calibration_date_count:].tolist())
    meta_train_optuna = pd.DataFrame(
        meta_train_panel.loc[~meta_train_panel[DATE_COLUMN].isin(calibration_dates)].copy(),
    )
    meta_train_calibration = pd.DataFrame(
        meta_train_panel.loc[meta_train_panel[DATE_COLUMN].isin(calibration_dates)].copy(),
    )
    LOGGER.info(
        "Calibration holdout split: optuna_dates=%d | calibration_dates=%d (%.1f%%)",
        len(unique_train_dates) - calibration_date_count,
        calibration_date_count,
        100.0 * runtime.calibration_holdout_fraction,
    )
    trial_ledger, best_params = run_meta_optuna_study(
        meta_train_optuna,
        meta_feature_columns=meta_feature_columns,
        runtime=runtime,
    )
    meta_train_oof = _build_meta_oof_predictions(
        meta_train_optuna,
        meta_feature_columns=meta_feature_columns,
        runtime=runtime,
        best_params=best_params,
    )
    calibration_artifact, calibration_scored = _fit_final_meta_outputs(
        meta_train_optuna,
        meta_train_calibration,
        meta_feature_columns=meta_feature_columns,
        runtime=runtime,
        best_params=best_params,
    )
    calibration_candidates = select_meta_candidate_rows(calibration_scored)
    calibrator = fit_probability_calibrator_train_only(
        calibration_candidates,
        probability_column=META_PROBABILITY_COLUMN,
        label_column=META_LABEL_COLUMN,
    )
    meta_train_candidate_index = select_meta_candidate_rows(meta_train_oof).index
    meta_train_candidate_probabilities = meta_train_oof.loc[
        meta_train_candidate_index,
        META_PROBABILITY_COLUMN,
    ].to_numpy(dtype=np.float64)
    meta_train_oof.loc[:, META_PROBABILITY_COLUMN] = 0.0
    if len(meta_train_candidate_index) > 0:
        meta_train_oof.loc[meta_train_candidate_index, META_PROBABILITY_COLUMN] = calibrator.transform(
            meta_train_candidate_probabilities,
        )
    meta_train_oof = attach_refined_signal_columns(
        meta_train_oof,
        strategy=runtime.refinement_strategy,
        soft_shifted_floor=runtime.soft_shifted_floor,
        rank_blend_lambda=runtime.rank_blend_lambda,
    )
    meta_model_payload, meta_val_predictions = _fit_final_meta_outputs(
        meta_train_panel,
        meta_val_panel,
        meta_feature_columns=meta_feature_columns,
        runtime=runtime,
        best_params=best_params,
    )
    meta_val_candidate_index = select_meta_candidate_rows(meta_val_predictions).index
    meta_val_candidate_probabilities = meta_val_predictions.loc[
        meta_val_candidate_index,
        META_PROBABILITY_COLUMN,
    ].to_numpy(dtype=np.float64)
    meta_val_predictions.loc[:, META_PROBABILITY_COLUMN] = 0.0
    if len(meta_val_candidate_index) > 0:
        meta_val_predictions.loc[meta_val_candidate_index, META_PROBABILITY_COLUMN] = calibrator.transform(
            meta_val_candidate_probabilities,
        )
    meta_val_predictions = attach_refined_signal_columns(
        meta_val_predictions,
        strategy=runtime.refinement_strategy,
        soft_shifted_floor=runtime.soft_shifted_floor,
        rank_blend_lambda=runtime.rank_blend_lambda,
    )
    best_params_payload: dict[str, object] = {
        **best_params,
        "burn_in_train_fraction": float(runtime.burn_in_train_fraction),
        "boost_rounds": int(runtime.boost_rounds),
        "early_stopping_rounds": int(runtime.early_stopping_rounds),
        "early_stopping_validation_fraction": float(runtime.early_stopping_validation_fraction),
        "minimum_training_rounds": int(runtime.minimum_training_rounds),
        "meta_feature_columns": meta_feature_columns,
        "meta_probability_calibrator": serialize_probability_calibrator(calibrator),
        "refinement_strategy": runtime.refinement_strategy,
        "soft_shifted_floor": float(runtime.soft_shifted_floor),
        "rank_blend_lambda": float(runtime.rank_blend_lambda),
    }
    summary: dict[str, object] = {
        "train_tail_rows": int(len(meta_train_oof)),
        "val_rows": int(len(meta_val_predictions)),
        "objective_score_best": float(trial_ledger["objective_score"].max()),
    }
    save_meta_labeling_outputs(
        primary_oos_train_tail=primary_train_tail,
        primary_oos_val=primary_val,
        meta_train_oof_predictions=meta_train_oof,
        meta_val_predictions=meta_val_predictions,
        meta_trial_ledger=trial_ledger,
        meta_best_params=best_params_payload,
        meta_model_payload=meta_model_payload,
        meta_stage_summary=summary,
    )
    pipeline_elapsed: float = time.perf_counter() - pipeline_started_at
    LOGGER.info(
        "Meta-labeling pipeline completed: train_oof=%d | val=%d | best_score=%.6f | elapsed=%.1fs",
        int(summary["train_tail_rows"]),
        int(summary["val_rows"]),
        float(summary["objective_score_best"]),
        pipeline_elapsed,
    )
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_meta_labeling_pipeline()


if __name__ == "__main__":
    main()
