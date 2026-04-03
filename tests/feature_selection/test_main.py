from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.paths import FEATURE_SELECTION_FILTERED_DATASET_PARQUET
from core.src.meta_model.evaluate.dataset import load_preprocessed_evaluation_dataset
from core.src.meta_model.feature_selection.config import FeatureSelectionConfig
from core.src.meta_model.feature_selection.io import build_feature_selection_output_bundle
from core.src.meta_model.feature_selection.main import (
    build_selection_feature_columns,
    run_feature_selection,
)
from core.src.meta_model.feature_selection.config import FeatureSelectionConfig as SelectionConfig
from core.src.meta_model.feature_selection.reporting import validate_filtered_dataset_matches_selection
from core.src.meta_model.feature_selection.selection_pipeline import RobustFeatureSelectionResult
from core.src.meta_model.model_contract import (
    BENCHMARK_FORWARD_RETURN_COLUMN,
    CS_RANK_TARGET_COLUMN,
    CS_ZSCORE_TARGET_COLUMN,
    EXCESS_FORWARD_RETURN_COLUMN,
    MODEL_TARGET_COLUMN,
    RAW_FORWARD_RETURN_COLUMN,
)
from core.src.meta_model.optimize_parameters.dataset import load_preprocessed_dataset
from core.src.meta_model.optimize_parameters.main import optimize_xgboost_parameters


def _make_preprocessed_dataset() -> pd.DataFrame:
    date_index = pd.date_range("2014-01-02", periods=36, freq="B")
    dates = [cast(pd.Timestamp, date_index[index]) for index in range(len(date_index))]
    tickers = ("AAA", "BBB", "CCC")
    split_by_date: dict[pd.Timestamp, str] = {}
    for index, date in enumerate(dates):
        split_by_date[date] = "train" if index < 20 else "val" if index < 28 else "test"
    target_map = {"AAA": -1.0, "BBB": 0.0, "CCC": 1.0}
    rows: list[dict[str, object]] = []
    for date_position, current_date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            target_value = target_map[ticker]
            rows.append(
                {
                    "date": current_date,
                    "ticker": ticker,
                    "dataset_split": split_by_date[current_date],
                    "target_main": target_value,
                    RAW_FORWARD_RETURN_COLUMN: target_value,
                    BENCHMARK_FORWARD_RETURN_COLUMN: 0.0,
                    EXCESS_FORWARD_RETURN_COLUMN: target_value,
                    CS_ZSCORE_TARGET_COLUMN: target_value,
                    CS_RANK_TARGET_COLUMN: float(ticker_index + 1),
                    "feature_signal": target_value + (0.01 * date_position),
                    "feature_duplicate": target_value + (0.05 * date_position),
                    "feature_noise": float(date_position),
                    "feature_low_coverage": (
                        np.nan
                        if split_by_date[current_date] == "train"
                        and ticker == "AAA"
                        and date_position % 2 == 0
                        else target_value
                    ),
                    "feature_constant": 1.0,
                    "company_sector": "Technology" if ticker != "CCC" else "Industrials",
                    "company_industry": "Software" if ticker != "CCC" else "Machinery",
                    "stock_open_price": 100.0 + target_value + date_position,
                    "stock_high_price": 101.0 + target_value + date_position,
                    "stock_low_price": 99.0 + target_value + date_position,
                    "stock_close_price": 100.5 + target_value + date_position,
                    "stock_trading_volume": 1_000_000.0 + (10_000.0 * date_position) + ticker_index,
                },
            )
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def _make_mock_selection_result() -> RobustFeatureSelectionResult:
    score_frame = pd.DataFrame(
        {
            "feature_name": ["feature_signal", "feature_duplicate", "feature_noise"],
            "feature_family": ["other", "other", "other"],
            "feature_stem": ["feature_signal", "feature_duplicate", "feature_noise"],
            "objective_score": [0.05, 0.04, -0.01],
            "daily_rank_ic_mean": [0.05, 0.04, -0.01],
            "coverage_fraction": [1.0, 1.0, 1.0],
            "selected": [True, True, False],
            "selection_rank": [1, 2, 0],
            "drop_reason": ["selected", "selected", "non_positive_mda"],
        },
    )
    group_manifest = pd.DataFrame(
        {
            "group_id": ["other:feature_signal:0:1", "other:feature_duplicate:0:1"],
            "feature_family": ["other", "other"],
            "feature_stem": ["feature_signal", "feature_duplicate"],
            "group_level": [0, 0],
            "parent_group_id": [None, None],
            "feature_name": ["feature_signal", "feature_duplicate"],
        },
    )
    return RobustFeatureSelectionResult(
        score_frame=score_frame,
        selected_feature_names=["feature_signal", "feature_duplicate"],
        group_manifest=group_manifest,
        sfi_scores=score_frame,
        linear_pruning_audit=pd.DataFrame({"feature_name": ["feature_signal"]}),
        distance_correlation_audit=pd.DataFrame({"feature_name": ["feature_signal"]}),
        target_correlation_audit=pd.DataFrame({"feature_name": ["feature_signal"]}),
        mda_group_scores=pd.DataFrame({"feature_family": ["other"], "feature_stem": ["feature_signal"]}),
        mda_final_scores=score_frame,
        summary={"selected_feature_count": 2},
    )


class TestFeatureSelection:
    def test_feature_selection_config_defaults_to_all_detected_cores(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "core.src.meta_model.runtime_parallelism.os.cpu_count",
            lambda: 9,
        )

        config = SelectionConfig()

        assert config.parallel_workers == 9

    def test_build_selection_feature_columns_excludes_targets_and_metadata(self) -> None:
        dataset = _make_preprocessed_dataset()

        feature_columns = build_selection_feature_columns(dataset)

        assert "date" not in feature_columns
        assert "ticker" not in feature_columns
        assert "dataset_split" not in feature_columns
        assert MODEL_TARGET_COLUMN not in feature_columns
        assert "feature_signal" in feature_columns
        assert "feature_noise" in feature_columns

    def test_run_feature_selection_saves_new_stage_artifacts(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "preprocessed.parquet"
        output_bundle = build_feature_selection_output_bundle(
            tmp_path,
            filtered_dataset_parquet_name="filtered_dataset.parquet",
            filtered_dataset_csv_name="filtered_dataset.csv",
        )
        _make_preprocessed_dataset().to_parquet(dataset_path, index=False)

        with patch(
            "core.src.meta_model.feature_selection.main.run_robust_feature_selection",
            return_value=_make_mock_selection_result(),
        ):
            score_frame, selected_frame, filtered_dataset = run_feature_selection(
                dataset_path,
                FeatureSelectionConfig(fold_count=4, emit_input_inventory=False),
                output_bundle=output_bundle,
            )

        selected_feature_names = [
            str(feature_name)
            for feature_name in cast(list[object], selected_frame["feature_name"].tolist())
        ]
        assert selected_feature_names == ["feature_signal", "feature_duplicate"]
        assert output_bundle.filtered_dataset_parquet.exists()
        assert output_bundle.stability_scores_parquet.exists()
        assert output_bundle.selected_features_parquet.exists()
        assert output_bundle.sfi_scores_parquet.exists()
        assert output_bundle.linear_pruning_audit_parquet.exists()
        assert output_bundle.distance_correlation_audit_parquet.exists()
        assert output_bundle.target_correlation_audit_parquet.exists()
        assert output_bundle.mda_group_scores_parquet.exists()
        assert output_bundle.mda_final_scores_parquet.exists()
        assert {"feature_signal", "feature_duplicate"} <= set(filtered_dataset.columns)
        assert "hl_context_company_sector" in filtered_dataset.columns
        assert "hl_context_stock_open_price" in filtered_dataset.columns
        assert MODEL_TARGET_COLUMN in filtered_dataset.columns
        assert not score_frame.empty

    def test_optimize_and_evaluate_default_to_selected_dataset(self) -> None:
        optimize_dataset_default = inspect.signature(load_preprocessed_dataset).parameters["path"].default
        optimize_main_default = inspect.signature(optimize_xgboost_parameters).parameters["dataset_path"].default
        evaluate_dataset_default = inspect.signature(load_preprocessed_evaluation_dataset).parameters["path"].default

        assert optimize_dataset_default == FEATURE_SELECTION_FILTERED_DATASET_PARQUET
        assert optimize_main_default == FEATURE_SELECTION_FILTERED_DATASET_PARQUET
        assert evaluate_dataset_default == FEATURE_SELECTION_FILTERED_DATASET_PARQUET

    def test_run_feature_selection_raises_when_no_feature_survives(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "preprocessed.parquet"
        _make_preprocessed_dataset().to_parquet(dataset_path, index=False)
        empty_result = RobustFeatureSelectionResult(
            score_frame=pd.DataFrame(
                {
                    "feature_name": ["feature_signal"],
                    "objective_score": [0.01],
                    "daily_rank_ic_mean": [0.01],
                    "coverage_fraction": [1.0],
                    "mda_mean_delta_objective": [-0.001],
                    "drop_reason": ["non_positive_mda"],
                },
            ),
            selected_feature_names=[],
            group_manifest=pd.DataFrame(),
            sfi_scores=pd.DataFrame(),
            linear_pruning_audit=pd.DataFrame(),
            distance_correlation_audit=pd.DataFrame(),
            target_correlation_audit=pd.DataFrame(),
            mda_group_scores=pd.DataFrame(),
            mda_final_scores=pd.DataFrame(),
            summary={"selected_feature_count": 0},
        )

        with patch(
            "core.src.meta_model.feature_selection.main.run_robust_feature_selection",
            return_value=empty_result,
        ):
            with pytest.raises(RuntimeError, match="did not retain any feature"):
                run_feature_selection(dataset_path, FeatureSelectionConfig(fold_count=4, emit_input_inventory=False))

    def test_validate_filtered_dataset_matches_selection_detects_schema_drift(self) -> None:
        filtered_dataset = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02"]),
                "ticker": ["AAA"],
                "dataset_split": ["train"],
                MODEL_TARGET_COLUMN: [0.1],
                "feature_a": [1.0],
                "feature_b": [2.0],
            },
        )

        with pytest.raises(ValueError, match="Feature schema mismatch"):
            validate_filtered_dataset_matches_selection(filtered_dataset, selected_feature_names=["feature_a"])

    def test_validate_filtered_dataset_matches_selection_preserves_selection_order(self) -> None:
        filtered_dataset = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02"]),
                "ticker": ["AAA"],
                "dataset_split": ["train"],
                MODEL_TARGET_COLUMN: [0.1],
                "feature_b": [2.0],
                "feature_a": [1.0],
            },
        )

        validate_filtered_dataset_matches_selection(
            filtered_dataset,
            selected_feature_names=["feature_b", "feature_a"],
        )

    def test_run_feature_selection_accepts_selected_context_feature(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "preprocessed.parquet"
        _make_preprocessed_dataset().to_parquet(dataset_path, index=False)
        result = RobustFeatureSelectionResult(
            score_frame=pd.DataFrame(
                {
                    "feature_name": ["feature_signal", "stock_open_price"],
                    "objective_score": [0.02, 0.01],
                    "daily_rank_ic_mean": [0.02, 0.01],
                    "coverage_fraction": [1.0, 1.0],
                    "selected": [True, True],
                    "selection_rank": [1, 2],
                    "drop_reason": ["selected", "selected"],
                },
            ),
            selected_feature_names=["feature_signal", "stock_open_price"],
            group_manifest=pd.DataFrame(),
            sfi_scores=pd.DataFrame(),
            linear_pruning_audit=pd.DataFrame(),
            distance_correlation_audit=pd.DataFrame(),
            target_correlation_audit=pd.DataFrame(),
            mda_group_scores=pd.DataFrame(),
            mda_final_scores=pd.DataFrame(
                {
                    "feature_name": ["feature_signal", "stock_open_price"],
                    "mda_mean_delta_objective": [0.01, 0.005],
                    "mda_std_delta_objective": [0.0, 0.0],
                    "mda_fold_positive_share": [1.0, 1.0],
                    "mda_repeat_count": [4, 4],
                    "selected": [True, True],
                    "selection_rank": [1, 2],
                    "drop_reason": ["selected", "selected"],
                },
            ),
            summary={"selected_feature_count": 2},
        )

        with patch(
            "core.src.meta_model.feature_selection.main.run_robust_feature_selection",
            return_value=result,
        ):
            score_frame, selected_frame, filtered_dataset = run_feature_selection(
                dataset_path,
                FeatureSelectionConfig(fold_count=4, emit_input_inventory=False),
            )

        assert score_frame is not None
        assert selected_frame["feature_name"].tolist() == ["feature_signal", "stock_open_price"]
        assert "stock_open_price" in filtered_dataset.columns
        assert "hl_context_stock_open_price" in filtered_dataset.columns

    def test_run_feature_selection_uses_twenty_percent_of_train_rows_for_selection(
        self,
        tmp_path: Path,
    ) -> None:
        dataset_path = tmp_path / "preprocessed.parquet"
        _make_preprocessed_dataset().to_parquet(dataset_path, index=False)
        captured: dict[str, int] = {}

        def _fake_run_robust_feature_selection(cache, folds, feature_names, config):
            del folds, feature_names, config
            captured["train_row_count"] = cache.train_row_count
            return _make_mock_selection_result()

        with patch(
            "core.src.meta_model.feature_selection.main.run_robust_feature_selection",
            side_effect=_fake_run_robust_feature_selection,
        ):
            run_feature_selection(
                dataset_path,
                FeatureSelectionConfig(
                    fold_count=1,
                    train_sampling_fraction=0.20,
                    emit_input_inventory=False,
                ),
            )

        assert captured["train_row_count"] == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
