from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.evaluate.parameters import SelectedXGBoostConfiguration
from core.src.secondary_model.data.targets import SECONDARY_TARGET_SPECS
from core.src.secondary_model.oos_predictions.main import (
    build_secondary_prediction_column_name,
    generate_secondary_target_oos_predictions,
    run_secondary_oos_predictions,
)


def _make_secondary_preprocessed_df() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=10)
    split_labels = ["train"] * 4 + ["val"] * 2 + ["test"] * 4
    rows: list[dict[str, object]] = []
    for index, (current_date, split_name) in enumerate(zip(dates, split_labels, strict=True), start=1):
        rows.append({
            "date": current_date,
            "ticker": "AAA",
            "dataset_split": split_name,
            "target_main": float(index) / 100.0,
            "feature_a": float(index),
        })
    return pd.DataFrame(rows)


class TestGenerateSecondaryTargetOOSPredictions:
    def test_predicts_val_and_test_dates_with_label_embargo(self) -> None:
        data = _make_secondary_preprocessed_df()
        trained_max_dates: list[pd.Timestamp] = []

        def fake_train_final_xgboost_model(
            training_frame: pd.DataFrame,
            feature_columns: list[str],
            tuned_params: dict[str, Any],
            *,
            num_boost_round: int,
        ) -> object:
            del feature_columns, tuned_params, num_boost_round
            max_training_date = cast(pd.Timestamp, pd.to_datetime(training_frame["date"]).max())
            assert not pd.isna(max_training_date)
            trained_max_dates.append(cast(pd.Timestamp, max_training_date))
            return object()

        def fake_predict_test_frame(
            booster: object,
            test_frame: pd.DataFrame,
            feature_columns: list[str],
        ) -> pd.DataFrame:
            del booster, feature_columns
            predicted = test_frame.copy()
            predicted["prediction"] = np.arange(len(predicted), dtype=float)
            return predicted

        with (
            patch(
                "core.src.secondary_model.oos_predictions.main.train_final_xgboost_model",
                side_effect=fake_train_final_xgboost_model,
            ),
            patch(
                "core.src.secondary_model.oos_predictions.main.predict_test_frame",
                side_effect=fake_predict_test_frame,
            ),
        ):
            predictions = generate_secondary_target_oos_predictions(
                data,
                feature_columns=["feature_a"],
                tuned_params={"eta": 0.03},
                num_boost_round=10,
                hold_period_days=2,
                prediction_column_name="pred_future_trend_5d",
                logger=None,
            )

        expected_prediction_dates = pd.bdate_range("2020-01-07", periods=6)
        assert list(pd.to_datetime(predictions["date"]).tolist()) == list(expected_prediction_dates)
        assert "pred_future_trend_5d" in predictions.columns
        assert "prediction" not in predictions.columns
        assert trained_max_dates[0] == pd.Timestamp("2020-01-03")
        assert trained_max_dates[1] == pd.Timestamp("2020-01-06")


class TestRunSecondaryOOSPredictions:
    def test_runs_all_targets_merges_predictions_and_skips_existing_target(
        self,
        tmp_path: Path,
    ) -> None:
        preprocessed_dir = tmp_path / "preprocessed"
        oos_dir = tmp_path / "oos"
        skipped_target = SECONDARY_TARGET_SPECS[0].name

        for target_spec in SECONDARY_TARGET_SPECS:
            target_path = preprocessed_dir / target_spec.name / "dataset_preprocessed.parquet"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            _make_secondary_preprocessed_df().to_parquet(target_path, index=False)

        def target_preprocessed_path(target_name: str) -> Path:
            return preprocessed_dir / target_name / "dataset_preprocessed.parquet"

        def target_oos_dir(target_name: str) -> Path:
            return oos_dir / target_name

        def target_oos_parquet(target_name: str) -> Path:
            return target_oos_dir(target_name) / "dataset_oos_predictions.parquet"

        def target_oos_csv(target_name: str) -> Path:
            return target_oos_dir(target_name) / "dataset_oos_predictions_sample_5pct.csv"

        merged_parquet = oos_dir / "dataset_oos_predictions.parquet"
        merged_csv = oos_dir / "dataset_oos_predictions_sample_5pct.csv"

        existing_predictions = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-07", "2020-01-08"]),
            "ticker": ["AAA", "AAA"],
            "dataset_split": ["val", "val"],
            build_secondary_prediction_column_name(skipped_target): [0.1, 0.2],
        })
        target_oos_parquet(skipped_target).parent.mkdir(parents=True, exist_ok=True)
        existing_predictions.to_parquet(target_oos_parquet(skipped_target), index=False)
        existing_predictions.to_csv(target_oos_csv(skipped_target), index=False)

        generated_targets: list[str] = []

        def fake_generate_secondary_target_oos_predictions(
            data: pd.DataFrame,
            feature_columns: list[str],
            tuned_params: dict[str, Any],
            *,
            num_boost_round: int,
            hold_period_days: int,
            prediction_column_name: str,
            logger: Any,
        ) -> pd.DataFrame:
            del data, feature_columns, tuned_params, num_boost_round, hold_period_days, logger
            generated_targets.append(prediction_column_name.removeprefix("pred_"))
            return pd.DataFrame({
                "date": pd.to_datetime(["2020-01-07", "2020-01-08"]),
                "ticker": ["AAA", "AAA"],
                "dataset_split": ["val", "val"],
                prediction_column_name: [1.0, 2.0],
            })

        with (
            patch(
                "core.src.secondary_model.oos_predictions.main.build_secondary_preprocessed_dataset_parquet",
                side_effect=target_preprocessed_path,
            ),
            patch(
                "core.src.secondary_model.oos_predictions.main.build_secondary_target_oos_predictions_parquet",
                side_effect=target_oos_parquet,
            ),
            patch(
                "core.src.secondary_model.oos_predictions.main.build_secondary_target_oos_predictions_csv",
                side_effect=target_oos_csv,
            ),
            patch(
                "core.src.secondary_model.oos_predictions.main.SECONDARY_OOS_PREDICTIONS_PARQUET",
                merged_parquet,
            ),
            patch(
                "core.src.secondary_model.oos_predictions.main.SECONDARY_OOS_PREDICTIONS_SAMPLE_CSV",
                merged_csv,
            ),
            patch(
                "core.src.secondary_model.oos_predictions.main.load_selected_xgboost_configuration",
                return_value=SelectedXGBoostConfiguration(
                    selected_trial_number=0,
                    params={"eta": 0.03},
                    training_rounds=10,
                ),
            ),
            patch(
                "core.src.secondary_model.oos_predictions.main.build_feature_columns",
                return_value=["feature_a"],
            ),
            patch(
                "core.src.secondary_model.oos_predictions.main.generate_secondary_target_oos_predictions",
                side_effect=fake_generate_secondary_target_oos_predictions,
            ),
        ):
            merged = run_secondary_oos_predictions()

        assert skipped_target not in generated_targets
        assert set(generated_targets) == {
            target_spec.name for target_spec in SECONDARY_TARGET_SPECS[1:]
        }
        assert merged_parquet.exists()
        assert merged_csv.exists()
        expected_prediction_columns = {
            build_secondary_prediction_column_name(target_spec.name)
            for target_spec in SECONDARY_TARGET_SPECS
        }
        assert expected_prediction_columns.issubset(set(merged.columns))
