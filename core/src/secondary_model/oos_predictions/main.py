from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from pandas._libs.tslibs.nattype import NaTType

PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.constants import SAMPLE_FRAC
from core.src.meta_model.evaluate.dataset import (
    build_feature_columns,
    load_preprocessed_evaluation_dataset,
)
from core.src.meta_model.evaluate.parameters import load_selected_xgboost_configuration
from core.src.meta_model.evaluate.training import (
    build_available_training_frame,
    predict_test_frame,
    train_final_xgboost_model,
)
from core.src.secondary_model.data.paths import (
    SECONDARY_OOS_PREDICTIONS_DIR,
    SECONDARY_OOS_PREDICTIONS_PARQUET,
    SECONDARY_OOS_PREDICTIONS_SAMPLE_CSV,
    build_secondary_best_params_json,
    build_secondary_optuna_trials_parquet,
    build_secondary_preprocessed_dataset_parquet,
    build_secondary_target_oos_predictions_csv,
    build_secondary_target_oos_predictions_parquet,
)
from core.src.secondary_model.data.targets import SECONDARY_TARGET_SPECS, TARGET_HORIZON_DAYS

LOGGER: logging.Logger = logging.getLogger(__name__)

DATE_COLUMN: str = "date"
TICKER_COLUMN: str = "ticker"
SPLIT_COLUMN: str = "dataset_split"
TRAIN_SPLIT_NAME: str = "train"
PREDICTION_SPLITS: tuple[str, ...] = ("val", "test")


def _format_duration(seconds: float) -> str:
    rounded_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(rounded_seconds, 60)
    hours, mins = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{mins:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def build_secondary_prediction_column_name(target_name: str) -> str:
    return f"pred_{target_name}"


def _load_existing_target_predictions(target_name: str) -> pd.DataFrame | None:
    target_predictions_path = build_secondary_target_oos_predictions_parquet(target_name)
    if not target_predictions_path.exists():
        return None
    existing = pd.read_parquet(target_predictions_path)
    existing[DATE_COLUMN] = pd.to_datetime(existing[DATE_COLUMN])
    return existing.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)


def _save_target_predictions(target_name: str, predictions: pd.DataFrame) -> None:
    target_predictions_path = build_secondary_target_oos_predictions_parquet(target_name)
    target_predictions_csv_path = build_secondary_target_oos_predictions_csv(target_name)
    target_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(target_predictions_path, index=False)
    predictions.sample(frac=min(1.0, SAMPLE_FRAC), random_state=7).to_csv(
        target_predictions_csv_path,
        index=False,
    )


def _save_merged_predictions(predictions: pd.DataFrame) -> None:
    SECONDARY_OOS_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(SECONDARY_OOS_PREDICTIONS_PARQUET, index=False)
    predictions.sample(frac=min(1.0, SAMPLE_FRAC), random_state=7).to_csv(
        SECONDARY_OOS_PREDICTIONS_SAMPLE_CSV,
        index=False,
    )


def generate_secondary_target_oos_predictions(
    data: pd.DataFrame,
    feature_columns: list[str],
    tuned_params: dict[str, Any],
    *,
    num_boost_round: int,
    hold_period_days: int,
    prediction_column_name: str,
    logger: Any,
) -> pd.DataFrame:
    ordered = data.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
    prediction_mask = ordered[SPLIT_COLUMN].isin(PREDICTION_SPLITS)
    prediction_dates = pd.Index(
        pd.to_datetime(ordered.loc[prediction_mask, DATE_COLUMN]).drop_duplicates().sort_values(),
    )
    prediction_parts: list[pd.DataFrame] = []
    started_at = time.perf_counter()
    total_dates = len(prediction_dates)
    if total_dates == 0:
        raise ValueError("No OOS prediction dates found for secondary model pipeline.")

    for date_index, prediction_date in enumerate(prediction_dates, start=1):
        prediction_timestamp_or_nat = pd.Timestamp(prediction_date)
        if isinstance(prediction_timestamp_or_nat, NaTType):
            raise ValueError("Prediction date cannot be NaT in secondary OOS pipeline.")
        safe_prediction_timestamp = prediction_timestamp_or_nat
        available_training_frame = build_available_training_frame(
            ordered,
            prediction_date=safe_prediction_timestamp,
            hold_period_days=hold_period_days,
        )
        prediction_frame = pd.DataFrame(
            ordered.loc[prediction_mask & (ordered[DATE_COLUMN] == safe_prediction_timestamp)].copy(),
        )
        booster = train_final_xgboost_model(
            available_training_frame,
            feature_columns,
            tuned_params,
            num_boost_round=num_boost_round,
        )
        predicted_frame = predict_test_frame(booster, prediction_frame, feature_columns)
        predicted_frame = predicted_frame.rename(columns={"prediction": prediction_column_name})
        prediction_parts.append(predicted_frame.loc[:, [DATE_COLUMN, TICKER_COLUMN, SPLIT_COLUMN, prediction_column_name]])
        if logger is not None and (
            date_index == 1 or date_index % 20 == 0 or date_index == total_dates
        ):
            elapsed_seconds = time.perf_counter() - started_at
            average_seconds = elapsed_seconds / date_index
            remaining_dates = total_dates - date_index
            logger.info(
                (
                    "Secondary OOS walk-forward: %d/%d dates (%.2f%%) | "
                    "prediction_date=%s | train_rows=%d | predict_rows=%d | "
                    "elapsed=%s | avg/date=%.2fs | eta=%s"
                ),
                date_index,
                total_dates,
                100.0 * date_index / total_dates,
                safe_prediction_timestamp.date(),
                len(available_training_frame),
                len(prediction_frame),
                _format_duration(elapsed_seconds),
                average_seconds,
                _format_duration(average_seconds * remaining_dates),
            )
    return pd.concat(prediction_parts, ignore_index=True)


def _merge_prediction_frames(target_prediction_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for target_spec in SECONDARY_TARGET_SPECS:
        prediction_column = build_secondary_prediction_column_name(target_spec.name)
        current = pd.DataFrame(
            target_prediction_frames[target_spec.name].loc[
                :,
                [DATE_COLUMN, TICKER_COLUMN, SPLIT_COLUMN, prediction_column],
            ].copy(),
        )
        if merged is None:
            merged = current
            continue
        merged = pd.DataFrame(
            merged.merge(
                current,
                on=[DATE_COLUMN, TICKER_COLUMN, SPLIT_COLUMN],
                how="outer",
                validate="one_to_one",
            ),
        )
    if merged is None:
        raise ValueError("No secondary OOS prediction frames available to merge.")
    return merged.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)


def run_secondary_oos_predictions(
    hold_period_days: int = TARGET_HORIZON_DAYS,
) -> pd.DataFrame:
    target_prediction_frames: dict[str, pd.DataFrame] = {}
    SECONDARY_OOS_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    for target_spec in SECONDARY_TARGET_SPECS:
        prediction_column_name = build_secondary_prediction_column_name(target_spec.name)
        existing_predictions = _load_existing_target_predictions(target_spec.name)
        if existing_predictions is not None:
            LOGGER.info(
                "Skipping OOS prediction generation for %s because outputs already exist.",
                target_spec.name,
            )
            target_prediction_frames[target_spec.name] = existing_predictions
            continue

        dataset_path = build_secondary_preprocessed_dataset_parquet(target_spec.name)
        best_params_path = build_secondary_best_params_json(target_spec.name)
        data = load_preprocessed_evaluation_dataset(dataset_path)
        feature_columns = build_feature_columns(data)
        selected_configuration = load_selected_xgboost_configuration(
            best_params_path=best_params_path,
            trials_path=build_secondary_optuna_trials_parquet(target_spec.name),
        )
        LOGGER.info(
            "Starting secondary OOS predictions for %s: rows=%d | features=%d | selected_trial=%d | training_rounds=%d",
            target_spec.name,
            len(data),
            len(feature_columns),
            selected_configuration.selected_trial_number,
            selected_configuration.training_rounds,
        )
        predictions = generate_secondary_target_oos_predictions(
            data,
            feature_columns,
            selected_configuration.params,
            num_boost_round=selected_configuration.training_rounds,
            hold_period_days=hold_period_days,
            prediction_column_name=prediction_column_name,
            logger=LOGGER,
        )
        _save_target_predictions(target_spec.name, predictions)
        target_prediction_frames[target_spec.name] = predictions
        LOGGER.info(
            "Completed secondary OOS predictions for %s: %d rows.",
            target_spec.name,
            len(predictions),
        )

    merged_predictions = _merge_prediction_frames(target_prediction_frames)
    _save_merged_predictions(merged_predictions)
    LOGGER.info(
        "Secondary OOS prediction pipeline completed: %d rows x %d cols.",
        len(merged_predictions),
        len(merged_predictions.columns),
    )
    return merged_predictions


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_secondary_oos_predictions()


__all__ = [
    "build_secondary_prediction_column_name",
    "generate_secondary_target_oos_predictions",
    "main",
    "run_secondary_oos_predictions",
]


if __name__ == "__main__":
    main()
