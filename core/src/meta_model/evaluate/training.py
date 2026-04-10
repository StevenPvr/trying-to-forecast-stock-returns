from __future__ import annotations

"""Walk-forward retraining: build training frames, train models, and yield daily predictions."""

import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from core.src.meta_model.evaluate.config import TARGET_COLUMN
from core.src.meta_model.model_registry.main import ModelSpec, fit_model, predict_model
from core.src.meta_model.runtime_parallelism import resolve_available_cpu_count
from core.src.meta_model.model_contract import (
    DATE_COLUMN,
    PREDICTION_COLUMN,
    SIGNAL_DATE_COLUMN,
    SPLIT_COLUMN,
    TEST_SPLIT_NAME,
    TRAIN_SPLIT_NAME,
    TICKER_COLUMN,
)
from core.src.meta_model.optimize_parameters.search_space import load_xgboost_module
from core.src.meta_model.xgboost_dmatrix import build_xgboost_dmatrix, prepare_xgboost_feature_frame
from core.src.meta_model.split_guard import assert_train_only_fit_frame


def _format_duration(seconds: float) -> str:
    rounded_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(rounded_seconds, 60)
    hours, mins = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{mins:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def build_available_training_frame(
    data: pd.DataFrame,
    *,
    prediction_date: pd.Timestamp,
    label_embargo_days: int,
) -> pd.DataFrame:
    """Return training rows up to *prediction_date* minus the label embargo."""
    ordered = data.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
    unique_dates = pd.Index(pd.to_datetime(ordered[DATE_COLUMN]).drop_duplicates().sort_values())
    matching_positions = np.flatnonzero(unique_dates == prediction_date)
    if matching_positions.size == 0:
        raise ValueError(f"Prediction date {prediction_date} not found in evaluation dataset.")

    cutoff_position = int(matching_positions[0]) - label_embargo_days
    if cutoff_position < 0:
        raise ValueError(
            "Not enough historical dates to respect the label embargo before prediction.",
        )

    latest_label_date = pd.Timestamp(unique_dates.tolist()[cutoff_position])
    return pd.DataFrame(
        ordered.loc[ordered[DATE_COLUMN] <= latest_label_date].copy(),
    ).reset_index(drop=True)


def resolve_training_threads() -> int:
    return resolve_available_cpu_count()


def build_xgboost_training_params(
    tuned_params: dict[str, Any],
    *,
    random_seed: int = 7,
    nthread: int | None = None,
) -> dict[str, Any]:
    """Merge tuned hyper-params with base XGBoost defaults (hist, squarederror)."""
    params: dict[str, Any] = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "seed": random_seed,
        "verbosity": 0,
        "nthread": nthread or resolve_training_threads(),
    }
    params.update(tuned_params)
    return params


def train_final_xgboost_model(
    training_frame: pd.DataFrame,
    feature_columns: list[str],
    tuned_params: dict[str, Any],
    *,
    num_boost_round: int,
) -> Any:
    """Train a final XGBoost booster on the full *training_frame*."""
    xgb = load_xgboost_module()
    feature_slice = prepare_xgboost_feature_frame(training_frame, feature_columns)
    training_matrix = build_xgboost_dmatrix(
        xgb,
        feature_slice,
        training_frame[TARGET_COLUMN].to_numpy(dtype=np.float32),
    )
    return xgb.train(
        params=build_xgboost_training_params(tuned_params),
        dtrain=training_matrix,
        num_boost_round=num_boost_round,
        verbose_eval=False,
    )


def predict_test_frame(
    booster: Any,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    execution_date: pd.Timestamp,
) -> pd.DataFrame:
    """Score *test_frame* using a trained booster, tagging with *execution_date*."""
    xgb = load_xgboost_module()
    test_slice = prepare_xgboost_feature_frame(test_frame, feature_columns)
    test_matrix = build_xgboost_dmatrix(xgb, test_slice, label=None)
    predictions = np.asarray(booster.predict(test_matrix), dtype=np.float64)
    predicted = test_frame.copy()
    predicted[SIGNAL_DATE_COLUMN] = pd.to_datetime(predicted[DATE_COLUMN])
    predicted[DATE_COLUMN] = execution_date
    predicted[PREDICTION_COLUMN] = predictions
    return predicted


def predict_test_frame_from_values(
    test_frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    execution_date: pd.Timestamp,
) -> pd.DataFrame:
    predicted = test_frame.copy()
    predicted[SIGNAL_DATE_COLUMN] = pd.to_datetime(predicted[DATE_COLUMN])
    predicted[DATE_COLUMN] = execution_date
    predicted[PREDICTION_COLUMN] = np.asarray(predictions, dtype=np.float64)
    return predicted


def _resolve_execution_date(
    prediction_date: pd.Timestamp,
    test_dates: pd.Index,
    execution_lag_days: int,
) -> pd.Timestamp | None:
    matching_positions = np.flatnonzero(test_dates == prediction_date)
    if matching_positions.size == 0:
        raise ValueError(f"Prediction date {prediction_date} not found in test split.")
    execution_position = int(matching_positions[0]) + execution_lag_days
    if execution_position >= len(test_dates):
        return None
    return pd.Timestamp(test_dates.tolist()[execution_position])


def generate_walk_forward_predictions(
    data: pd.DataFrame,
    feature_columns: list[str],
    tuned_params: dict[str, Any],
    *,
    num_boost_round: int,
    hold_period_days: int,
    execution_lag_days: int,
    logger: Any,
) -> pd.DataFrame:
    ordered = data.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
    test_dates = pd.Index(
        pd.to_datetime(
            ordered.loc[ordered[SPLIT_COLUMN] == TEST_SPLIT_NAME, DATE_COLUMN],
        ).drop_duplicates().sort_values(),
    )
    prediction_parts: list[pd.DataFrame] = []
    started_at = time.perf_counter()
    for date_index, prediction_date in enumerate(test_dates, start=1):
        execution_date = _resolve_execution_date(
            pd.Timestamp(prediction_date),
            test_dates,
            execution_lag_days,
        )
        if execution_date is None:
            continue
        available_training_frame = build_available_training_frame(
            ordered,
            prediction_date=pd.Timestamp(prediction_date),
            label_embargo_days=hold_period_days + execution_lag_days,
        )
        prediction_frame = pd.DataFrame(
            ordered.loc[
                (ordered[SPLIT_COLUMN] == TEST_SPLIT_NAME)
                & (ordered[DATE_COLUMN] == prediction_date)
            ].copy(),
        )
        booster = train_final_xgboost_model(
            available_training_frame,
            feature_columns,
            tuned_params,
            num_boost_round=num_boost_round,
        )
        prediction_parts.append(
            predict_test_frame(
                booster,
                prediction_frame,
                feature_columns,
                execution_date=execution_date,
            ),
        )
        if date_index == 1 or date_index % 20 == 0 or date_index == len(test_dates):
            elapsed_seconds = time.perf_counter() - started_at
            average_seconds = elapsed_seconds / date_index
            remaining_dates = len(test_dates) - date_index
            eta_seconds = average_seconds * remaining_dates
            logger.info(
                "Walk-forward retrain: %d/%d test dates (%.2f%%) | signal_date=%s | execution_date=%s | train_rows=%d | predict_rows=%d | elapsed=%s | avg/date=%.2fs | eta=%s",
                date_index,
                len(test_dates),
                100.0 * date_index / len(test_dates),
                pd.Timestamp(prediction_date).date(),
                execution_date.date(),
                len(available_training_frame),
                len(prediction_frame),
                _format_duration(elapsed_seconds),
                average_seconds,
                _format_duration(eta_seconds),
            )
    if not prediction_parts:
        raise ValueError("No test predictions were generated.")
    return pd.concat(prediction_parts, ignore_index=True)


def iter_walk_forward_prediction_days(
    data: pd.DataFrame,
    feature_columns: list[str],
    tuned_params: dict[str, Any],
    *,
    num_boost_round: int,
    hold_period_days: int,
    execution_lag_days: int,
    logger: Any,
) -> Iterator[pd.DataFrame]:
    ordered = data.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
    test_dates = pd.Index(
        pd.to_datetime(
            ordered.loc[ordered[SPLIT_COLUMN] == TEST_SPLIT_NAME, DATE_COLUMN],
        ).drop_duplicates().sort_values(),
    )
    started_at = time.perf_counter()
    for date_index, prediction_date in enumerate(test_dates, start=1):
        execution_date = _resolve_execution_date(
            pd.Timestamp(prediction_date),
            test_dates,
            execution_lag_days,
        )
        if execution_date is None:
            continue
        available_training_frame = build_available_training_frame(
            ordered,
            prediction_date=pd.Timestamp(prediction_date),
            label_embargo_days=hold_period_days + execution_lag_days,
        )
        prediction_frame = pd.DataFrame(
            ordered.loc[
                (ordered[SPLIT_COLUMN] == TEST_SPLIT_NAME)
                & (ordered[DATE_COLUMN] == prediction_date)
            ].copy(),
        )
        booster = train_final_xgboost_model(
            available_training_frame,
            feature_columns,
            tuned_params,
            num_boost_round=num_boost_round,
        )
        predicted_frame = predict_test_frame(
            booster,
            prediction_frame,
            feature_columns,
            execution_date=execution_date,
        )
        elapsed_seconds = time.perf_counter() - started_at
        average_seconds = elapsed_seconds / date_index
        remaining_dates = len(test_dates) - date_index
        eta_seconds = average_seconds * remaining_dates
        logger.info(
            "Walk-forward retrain: %d/%d test dates (%.2f%%) | signal_date=%s | execution_date=%s | train_rows=%d | predict_rows=%d | elapsed=%s | avg/date=%.2fs | eta=%s",
            date_index,
            len(test_dates),
            100.0 * date_index / len(test_dates),
            pd.Timestamp(prediction_date).date(),
            execution_date.date(),
            len(available_training_frame),
            len(prediction_frame),
            _format_duration(elapsed_seconds),
            average_seconds,
            _format_duration(eta_seconds),
        )
        yield predicted_frame


def iter_model_prediction_days(
    data: pd.DataFrame,
    feature_columns: list[str],
    model_spec: ModelSpec,
    *,
    hold_period_days: int,
    execution_lag_days: int,
    logger: Any,
) -> Iterator[pd.DataFrame]:
    """Yield one predicted DataFrame per test date using walk-forward retraining."""
    ordered = data.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
    test_dates = pd.Index(
        pd.to_datetime(
            ordered.loc[ordered[SPLIT_COLUMN] == TEST_SPLIT_NAME, DATE_COLUMN],
        ).drop_duplicates().sort_values(),
    )
    started_at = time.perf_counter()
    for date_index, prediction_date in enumerate(test_dates, start=1):
        execution_date = _resolve_execution_date(
            pd.Timestamp(prediction_date),
            test_dates,
            execution_lag_days,
        )
        if execution_date is None:
            continue
        available_training_frame = build_available_training_frame(
            ordered,
            prediction_date=pd.Timestamp(prediction_date),
            label_embargo_days=hold_period_days + execution_lag_days,
        )
        prediction_frame = pd.DataFrame(
            ordered.loc[
                (ordered[SPLIT_COLUMN] == TEST_SPLIT_NAME)
                & (ordered[DATE_COLUMN] == prediction_date)
            ].copy(),
        )
        artifact = fit_model(model_spec, available_training_frame, feature_columns)
        predictions = predict_model(artifact, prediction_frame, feature_columns)
        predicted_frame = predict_test_frame_from_values(
            prediction_frame,
            predictions,
            execution_date=execution_date,
        )
        elapsed_seconds = time.perf_counter() - started_at
        average_seconds = elapsed_seconds / date_index
        remaining_dates = len(test_dates) - date_index
        eta_seconds = average_seconds * remaining_dates
        logger.info(
            "Walk-forward retrain [%s]: %d/%d test dates (%.2f%%) | signal_date=%s | execution_date=%s | train_rows=%d | predict_rows=%d | elapsed=%s | avg/date=%.2fs | eta=%s",
            model_spec.model_name,
            date_index,
            len(test_dates),
            100.0 * date_index / len(test_dates),
            pd.Timestamp(prediction_date).date(),
            execution_date.date(),
            len(available_training_frame),
            len(prediction_frame),
            _format_duration(elapsed_seconds),
            average_seconds,
            _format_duration(eta_seconds),
        )
        yield predicted_frame


def _resolve_train_and_test_frames(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = data.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
    train_frame = pd.DataFrame(ordered.loc[ordered[SPLIT_COLUMN] == TRAIN_SPLIT_NAME].copy())
    test_frame = pd.DataFrame(ordered.loc[ordered[SPLIT_COLUMN] == TEST_SPLIT_NAME].copy())
    if train_frame.empty:
        raise ValueError("No train rows available for frozen_train_only evaluation mode.")
    if test_frame.empty:
        raise ValueError("No test rows available for frozen_train_only evaluation mode.")
    assert_train_only_fit_frame(train_frame, context="evaluate.training.frozen_train_only")
    return train_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


def iter_model_prediction_days_frozen_train_only(
    data: pd.DataFrame,
    feature_columns: list[str],
    model_spec: ModelSpec,
    *,
    execution_lag_days: int,
    logger: Any,
) -> Iterator[pd.DataFrame]:
    """Yield prediction days with a single model fit on the train split only."""
    train_frame, test_frame = _resolve_train_and_test_frames(data)
    artifact = fit_model(model_spec, train_frame, feature_columns)
    test_dates = pd.Index(pd.to_datetime(test_frame[DATE_COLUMN]).drop_duplicates().sort_values())
    started_at = time.perf_counter()
    for date_index, prediction_date in enumerate(test_dates, start=1):
        execution_date = _resolve_execution_date(
            pd.Timestamp(prediction_date),
            test_dates,
            execution_lag_days,
        )
        if execution_date is None:
            continue
        prediction_frame = pd.DataFrame(
            test_frame.loc[test_frame[DATE_COLUMN] == prediction_date].copy(),
        )
        predictions = predict_model(artifact, prediction_frame, feature_columns)
        predicted_frame = predict_test_frame_from_values(
            prediction_frame,
            predictions,
            execution_date=execution_date,
        )
        elapsed_seconds = time.perf_counter() - started_at
        average_seconds = elapsed_seconds / date_index
        remaining_dates = len(test_dates) - date_index
        eta_seconds = average_seconds * remaining_dates
        logger.info(
            "Frozen train-only predict [%s]: %d/%d test dates (%.2f%%) | signal_date=%s | execution_date=%s | train_rows=%d | predict_rows=%d | elapsed=%s | avg/date=%.2fs | eta=%s",
            model_spec.model_name,
            date_index,
            len(test_dates),
            100.0 * date_index / len(test_dates),
            pd.Timestamp(prediction_date).date(),
            execution_date.date(),
            len(train_frame),
            len(prediction_frame),
            _format_duration(elapsed_seconds),
            average_seconds,
            _format_duration(eta_seconds),
        )
        yield predicted_frame
