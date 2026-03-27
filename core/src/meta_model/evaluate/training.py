from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any
from typing import cast

import numpy as np
import pandas as pd

from core.src.meta_model.evaluate.config import (
    DATE_COLUMN,
    PREDICTION_COLUMN,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    TEST_SPLIT_NAME,
    TICKER_COLUMN,
)
from core.src.meta_model.optimize_parameters.search_space import load_xgboost_module


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
    hold_period_days: int,
) -> pd.DataFrame:
    ordered = data.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
    unique_dates = pd.Index(pd.to_datetime(ordered[DATE_COLUMN]).drop_duplicates().sort_values())
    matching_positions = np.flatnonzero(unique_dates == prediction_date)
    if matching_positions.size == 0:
        raise ValueError(f"Prediction date {prediction_date} not found in evaluation dataset.")

    cutoff_position = int(matching_positions[0]) - hold_period_days
    if cutoff_position < 0:
        raise ValueError(
            "Not enough historical dates to respect the label embargo before prediction.",
        )

    latest_label_date = cast(pd.Timestamp, pd.Timestamp(unique_dates.tolist()[cutoff_position]))
    return pd.DataFrame(
        ordered.loc[ordered[DATE_COLUMN] <= latest_label_date].copy(),
    ).reset_index(drop=True)


def resolve_training_threads() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def build_xgboost_training_params(
    tuned_params: dict[str, Any],
    *,
    random_seed: int = 7,
    nthread: int | None = None,
) -> dict[str, Any]:
    params = {
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
    xgb = load_xgboost_module()
    training_matrix = xgb.DMatrix(
        training_frame.loc[:, feature_columns],
        label=training_frame[TARGET_COLUMN].to_numpy(dtype=np.float32),
        feature_names=feature_columns,
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
) -> pd.DataFrame:
    xgb = load_xgboost_module()
    test_matrix = xgb.DMatrix(
        test_frame.loc[:, feature_columns],
        feature_names=feature_columns,
    )
    predictions = np.asarray(booster.predict(test_matrix), dtype=np.float64)
    predicted = test_frame.copy()
    predicted[PREDICTION_COLUMN] = predictions
    return predicted


def generate_walk_forward_predictions(
    data: pd.DataFrame,
    feature_columns: list[str],
    tuned_params: dict[str, Any],
    *,
    num_boost_round: int,
    hold_period_days: int,
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
        available_training_frame = build_available_training_frame(
            ordered,
            prediction_date=cast(pd.Timestamp, pd.Timestamp(prediction_date)),
            hold_period_days=hold_period_days,
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
            predict_test_frame(booster, prediction_frame, feature_columns),
        )
        if date_index == 1 or date_index % 20 == 0 or date_index == len(test_dates):
            elapsed_seconds = time.perf_counter() - started_at
            average_seconds = elapsed_seconds / date_index
            remaining_dates = len(test_dates) - date_index
            eta_seconds = average_seconds * remaining_dates
            logger.info(
                "Walk-forward retrain: %d/%d test dates (%.2f%%) | prediction_date=%s | train_rows=%d | predict_rows=%d | elapsed=%s | avg/date=%.2fs | eta=%s",
                date_index,
                len(test_dates),
                100.0 * date_index / len(test_dates),
                pd.Timestamp(prediction_date).date(),
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
        available_training_frame = build_available_training_frame(
            ordered,
            prediction_date=cast(pd.Timestamp, pd.Timestamp(prediction_date)),
            hold_period_days=hold_period_days,
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
        predicted_frame = predict_test_frame(booster, prediction_frame, feature_columns)
        elapsed_seconds = time.perf_counter() - started_at
        average_seconds = elapsed_seconds / date_index
        remaining_dates = len(test_dates) - date_index
        eta_seconds = average_seconds * remaining_dates
        logger.info(
            "Walk-forward retrain: %d/%d test dates (%.2f%%) | prediction_date=%s | train_rows=%d | predict_rows=%d | elapsed=%s | avg/date=%.2fs | eta=%s",
            date_index,
            len(test_dates),
            100.0 * date_index / len(test_dates),
            pd.Timestamp(prediction_date).date(),
            len(available_training_frame),
            len(prediction_frame),
            _format_duration(elapsed_seconds),
            average_seconds,
            _format_duration(eta_seconds),
        )
        yield predicted_frame
