from __future__ import annotations

import sys
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.secondary_model.feature_selection.main import (
    SECONDARY_TARGET_SPECS,
    build_secondary_selection_scaffold,
    derive_secondary_train_end_date,
    run_secondary_feature_selection,
)
from core.src.secondary_model.data.data_preprocessing.main import (
    SECONDARY_TARGET_SPECS as PREPROCESSING_SECONDARY_TARGET_SPECS,
    assign_secondary_dataset_splits,
)
from core.src.meta_model.data.data_preprocessing.main import (
    TARGET_COLUMN,
    filter_from_start_date,
    remove_rows_with_missing_values,
)


def _make_feature_df() -> pd.DataFrame:
    dates = pd.bdate_range("2009-01-02", periods=3200)
    rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(("AAPL", "MSFT"), start=1):
        close_prices = 100.0 * ticker_index + np.arange(len(dates), dtype=float)
        volume = 1_000_000.0 + np.arange(len(dates), dtype=float) * 100.0
        for idx, current_date in enumerate(dates):
            rows.append(
                {
                    "date": current_date,
                    "ticker": ticker,
                    "stock_close_price": close_prices[idx],
                    "stock_trading_volume": volume[idx],
                    "ta_signal": float(idx),
                    "quant_signal": float(idx) / 2.0,
                },
            )
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


class TestSecondarySelectionScaffold:
    def test_feature_selection_and_preprocessing_share_exact_same_target_specs(self) -> None:
        assert SECONDARY_TARGET_SPECS is PREPROCESSING_SECONDARY_TARGET_SPECS

    def test_builds_secondary_splits_for_selection_scaffold(self, tmp_path: Path) -> None:
        feature_path = tmp_path / "features.parquet"
        _make_feature_df().to_parquet(feature_path, index=False)

        scaffold = build_secondary_selection_scaffold(feature_path, SECONDARY_TARGET_SPECS[0])

        assert {"row_position", "date", "ticker", "target_main", "dataset_split"} == set(scaffold.columns)
        assert set(scaffold["dataset_split"].unique()) == {"train", "val", "test"}

    def test_selection_scaffold_target_matches_direct_target_calculation_for_each_secondary_target(
        self,
        tmp_path: Path,
    ) -> None:
        feature_path = tmp_path / "features.parquet"
        dataset = _make_feature_df()
        dataset.to_parquet(feature_path, index=False)
        filtered = filter_from_start_date(dataset)

        for target_spec in SECONDARY_TARGET_SPECS:
            scaffold = build_secondary_selection_scaffold(feature_path, target_spec)
            targeted = target_spec.build_target(filtered)
            split_ready = assign_secondary_dataset_splits(targeted)
            expected = remove_rows_with_missing_values(split_ready, required_columns=[TARGET_COLUMN])
            expected_target = (
                expected.loc[:, ["date", "ticker", TARGET_COLUMN]]
                .sort_values(["date", "ticker"])
                .reset_index(drop=True)
            )
            actual_target = (
                scaffold.loc[:, ["date", "ticker", TARGET_COLUMN]]
                .sort_values(["date", "ticker"])
                .reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(actual_target, expected_target)


class TestSecondaryFeatureSelection:
    def test_derives_train_end_date_from_secondary_train_split(self, tmp_path: Path) -> None:
        feature_path = tmp_path / "features.parquet"
        _make_feature_df().to_parquet(feature_path, index=False)
        target_spec = SECONDARY_TARGET_SPECS[0]

        train_end_date = derive_secondary_train_end_date(feature_path, target_spec)
        scaffold = build_secondary_selection_scaffold(feature_path, target_spec)
        expected = pd.to_datetime(
            scaffold.loc[scaffold["dataset_split"] == "train", "date"],
        ).max()

        assert train_end_date == expected

    def test_runs_corr_pca_then_greedy_for_each_secondary_target(self, tmp_path: Path) -> None:
        feature_path = tmp_path / "features.parquet"
        _make_feature_df().to_parquet(feature_path, index=False)

        captured_corr_calls: list[dict[str, object]] = []
        captured_greedy_calls: list[dict[str, object]] = []

        def fake_run_feature_corr_pca(**kwargs):
            captured_corr_calls.append(kwargs)
            return None, {}

        def fake_run_greedy_forward_selection(**kwargs):
            captured_greedy_calls.append(kwargs)
            return pd.DataFrame(), pd.DataFrame()

        with (
            patch(
                "core.src.secondary_model.feature_selection.main.run_feature_corr_pca",
                side_effect=fake_run_feature_corr_pca,
            ),
            patch(
                "core.src.secondary_model.feature_selection.main.run_greedy_forward_selection",
                side_effect=fake_run_greedy_forward_selection,
            ),
            patch(
                "core.src.secondary_model.feature_selection.main._feature_corr_pca_outputs_exist",
                return_value=False,
            ),
            patch(
                "core.src.secondary_model.feature_selection.main._greedy_outputs_exist",
                return_value=False,
            ),
        ):
            output_paths = run_secondary_feature_selection(feature_path)

        assert set(output_paths.keys()) == {target_spec.name for target_spec in SECONDARY_TARGET_SPECS}
        assert len(captured_corr_calls) == len(SECONDARY_TARGET_SPECS)
        assert len(captured_greedy_calls) == len(SECONDARY_TARGET_SPECS)

        for target_spec, corr_call, greedy_call in zip(
            SECONDARY_TARGET_SPECS,
            captured_corr_calls,
            captured_greedy_calls,
            strict=True,
        ):
            assert corr_call["feature_parquet_path"] == feature_path
            assert target_spec.name in str(corr_call["output_parquet_path"])
            assert target_spec.name in str(corr_call["output_sample_csv_path"])
            assert target_spec.name in str(corr_call["output_mapping_json_path"])
            assert isinstance(corr_call["train_end_date"], pd.Timestamp)

            selection_scaffold = greedy_call["selection_scaffold"]
            assert isinstance(selection_scaffold, pd.DataFrame)
            assert set(selection_scaffold["dataset_split"].unique()) == {"train", "val", "test"}
            assert greedy_call["feature_parquet_path"] == corr_call["output_parquet_path"]
            assert target_spec.name in str(greedy_call["scores_parquet_path"])
            assert target_spec.name in str(greedy_call["scores_csv_path"])
            assert target_spec.name in str(greedy_call["selected_features_parquet_path"])
            assert target_spec.name in str(greedy_call["selected_features_csv_path"])
            assert target_spec.name in str(greedy_call["filtered_features_parquet_path"])
            assert target_spec.name in str(greedy_call["filtered_features_csv_path"])
            metadata_columns_to_keep = cast(tuple[str, ...], greedy_call["metadata_columns_to_keep"])
            assert set(metadata_columns_to_keep) == {
                "date",
                "ticker",
                *target_spec.required_metadata_columns,
            }

    def test_creates_target_output_directories_before_writing(self, tmp_path: Path) -> None:
        feature_path = tmp_path / "features.parquet"
        _make_feature_df().to_parquet(feature_path, index=False)

        def fake_run_feature_corr_pca(**kwargs):
            output_parquet_path = Path(str(kwargs["output_parquet_path"]))
            output_mapping_json_path = Path(str(kwargs["output_mapping_json_path"]))
            assert output_parquet_path.parent.exists()
            assert output_mapping_json_path.parent.exists()
            return None, {}

        def fake_run_greedy_forward_selection(**kwargs):
            filtered_features_parquet_path = Path(str(kwargs["filtered_features_parquet_path"]))
            scores_parquet_path = Path(str(kwargs["scores_parquet_path"]))
            assert filtered_features_parquet_path.parent.exists()
            assert scores_parquet_path.parent.exists()
            return pd.DataFrame(), pd.DataFrame()

        with (
            patch(
                "core.src.secondary_model.feature_selection.main.run_feature_corr_pca",
                side_effect=fake_run_feature_corr_pca,
            ),
            patch(
                "core.src.secondary_model.feature_selection.main.run_greedy_forward_selection",
                side_effect=fake_run_greedy_forward_selection,
            ),
        ):
            run_secondary_feature_selection(feature_path)

    def test_skips_target_when_greedy_outputs_already_exist(self, tmp_path: Path) -> None:
        feature_path = tmp_path / "features.parquet"
        _make_feature_df().to_parquet(feature_path, index=False)
        skipped_target = SECONDARY_TARGET_SPECS[0]

        base_dir = tmp_path / "feature_selection_outputs"

        def target_dir(target_name: str) -> Path:
            return base_dir / target_name

        def corr_parquet(target_name: str) -> Path:
            return target_dir(target_name) / "dataset_features_corr_pca.parquet"

        def corr_csv(target_name: str) -> Path:
            return target_dir(target_name) / "dataset_features_corr_pca_sample_5pct.csv"

        def corr_json(target_name: str) -> Path:
            return target_dir(target_name) / "feature_corr_pca_mapping.json"

        def greedy_scores_parquet(target_name: str) -> Path:
            return target_dir(target_name) / "feature_greedy_forward_selection_scores.parquet"

        def greedy_scores_csv(target_name: str) -> Path:
            return target_dir(target_name) / "feature_greedy_forward_selection_scores.csv"

        def greedy_selected_parquet(target_name: str) -> Path:
            return target_dir(target_name) / "feature_greedy_forward_selection_selected.parquet"

        def greedy_selected_csv(target_name: str) -> Path:
            return target_dir(target_name) / "feature_greedy_forward_selection_selected.csv"

        def greedy_filtered_parquet(target_name: str) -> Path:
            return target_dir(target_name) / "dataset_features_greedy_forward_selected.parquet"

        def greedy_filtered_csv(target_name: str) -> Path:
            return target_dir(target_name) / "dataset_features_greedy_forward_selected_sample_5pct.csv"

        for path in (
            corr_parquet(skipped_target.name),
            corr_csv(skipped_target.name),
            corr_json(skipped_target.name),
            greedy_scores_parquet(skipped_target.name),
            greedy_scores_csv(skipped_target.name),
            greedy_selected_parquet(skipped_target.name),
            greedy_selected_csv(skipped_target.name),
            greedy_filtered_parquet(skipped_target.name),
            greedy_filtered_csv(skipped_target.name),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("done", encoding="utf-8")

        captured_corr_targets: list[str] = []
        captured_greedy_targets: list[str] = []

        def fake_run_feature_corr_pca(**kwargs):
            captured_corr_targets.append(Path(str(kwargs["output_parquet_path"])).parent.name)
            return None, {}

        def fake_run_greedy_forward_selection(**kwargs):
            captured_greedy_targets.append(Path(str(kwargs["filtered_features_parquet_path"])).parent.name)
            return pd.DataFrame(), pd.DataFrame()

        with (
            patch("core.src.secondary_model.feature_selection.main.build_secondary_feature_selection_target_dir", side_effect=target_dir),
            patch("core.src.secondary_model.feature_selection.main.build_secondary_feature_corr_pca_output_parquet", side_effect=corr_parquet),
            patch("core.src.secondary_model.feature_selection.main.build_secondary_feature_corr_pca_output_sample_csv", side_effect=corr_csv),
            patch("core.src.secondary_model.feature_selection.main.build_secondary_feature_corr_pca_mapping_json", side_effect=corr_json),
            patch("core.src.secondary_model.feature_selection.main.build_secondary_greedy_scores_parquet", side_effect=greedy_scores_parquet),
            patch("core.src.secondary_model.feature_selection.main.build_secondary_greedy_scores_csv", side_effect=greedy_scores_csv),
            patch("core.src.secondary_model.feature_selection.main.build_secondary_greedy_selected_features_parquet", side_effect=greedy_selected_parquet),
            patch("core.src.secondary_model.feature_selection.main.build_secondary_greedy_selected_features_csv", side_effect=greedy_selected_csv),
            patch("core.src.secondary_model.feature_selection.main.build_secondary_greedy_filtered_features_parquet", side_effect=greedy_filtered_parquet),
            patch("core.src.secondary_model.feature_selection.main.build_secondary_greedy_filtered_features_csv", side_effect=greedy_filtered_csv),
            patch("core.src.secondary_model.feature_selection.main.run_feature_corr_pca", side_effect=fake_run_feature_corr_pca),
            patch("core.src.secondary_model.feature_selection.main.run_greedy_forward_selection", side_effect=fake_run_greedy_forward_selection),
        ):
            output_paths = run_secondary_feature_selection(feature_path)

        assert output_paths[skipped_target.name] == greedy_filtered_parquet(skipped_target.name)
        assert skipped_target.name not in captured_corr_targets
        assert skipped_target.name not in captured_greedy_targets

    def test_skips_corr_pca_when_its_outputs_already_exist_but_still_runs_greedy(self, tmp_path: Path) -> None:
        feature_path = tmp_path / "features.parquet"
        _make_feature_df().to_parquet(feature_path, index=False)
        resumed_target = SECONDARY_TARGET_SPECS[0]

        base_dir = tmp_path / "feature_selection_outputs"

        def target_dir(target_name: str) -> Path:
            return base_dir / target_name

        def corr_parquet(target_name: str) -> Path:
            return target_dir(target_name) / "dataset_features_corr_pca.parquet"

        def corr_csv(target_name: str) -> Path:
            return target_dir(target_name) / "dataset_features_corr_pca_sample_5pct.csv"

        def corr_json(target_name: str) -> Path:
            return target_dir(target_name) / "feature_corr_pca_mapping.json"

        def greedy_scores_parquet(target_name: str) -> Path:
            return target_dir(target_name) / "feature_greedy_forward_selection_scores.parquet"

        def greedy_scores_csv(target_name: str) -> Path:
            return target_dir(target_name) / "feature_greedy_forward_selection_scores.csv"

        def greedy_selected_parquet(target_name: str) -> Path:
            return target_dir(target_name) / "feature_greedy_forward_selection_selected.parquet"

        def greedy_selected_csv(target_name: str) -> Path:
            return target_dir(target_name) / "feature_greedy_forward_selection_selected.csv"

        def greedy_filtered_parquet(target_name: str) -> Path:
            return target_dir(target_name) / "dataset_features_greedy_forward_selected.parquet"

        def greedy_filtered_csv(target_name: str) -> Path:
            return target_dir(target_name) / "dataset_features_greedy_forward_selected_sample_5pct.csv"

        for path in (
            corr_parquet(resumed_target.name),
            corr_csv(resumed_target.name),
            corr_json(resumed_target.name),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("done", encoding="utf-8")

        captured_corr_targets: list[str] = []
        captured_greedy_targets: list[str] = []

        def fake_run_feature_corr_pca(**kwargs):
            captured_corr_targets.append(Path(str(kwargs["output_parquet_path"])).parent.name)
            return None, {}

        def fake_run_greedy_forward_selection(**kwargs):
            captured_greedy_targets.append(Path(str(kwargs["filtered_features_parquet_path"])).parent.name)
            return pd.DataFrame(), pd.DataFrame()

        with (
            patch("core.src.secondary_model.feature_selection.main.build_secondary_feature_selection_target_dir", side_effect=target_dir),
            patch("core.src.secondary_model.feature_selection.main.build_secondary_feature_corr_pca_output_parquet", side_effect=corr_parquet),
            patch("core.src.secondary_model.feature_selection.main.build_secondary_feature_corr_pca_output_sample_csv", side_effect=corr_csv),
            patch("core.src.secondary_model.feature_selection.main.build_secondary_feature_corr_pca_mapping_json", side_effect=corr_json),
            patch("core.src.secondary_model.feature_selection.main.build_secondary_greedy_scores_parquet", side_effect=greedy_scores_parquet),
            patch("core.src.secondary_model.feature_selection.main.build_secondary_greedy_scores_csv", side_effect=greedy_scores_csv),
            patch("core.src.secondary_model.feature_selection.main.build_secondary_greedy_selected_features_parquet", side_effect=greedy_selected_parquet),
            patch("core.src.secondary_model.feature_selection.main.build_secondary_greedy_selected_features_csv", side_effect=greedy_selected_csv),
            patch("core.src.secondary_model.feature_selection.main.build_secondary_greedy_filtered_features_parquet", side_effect=greedy_filtered_parquet),
            patch("core.src.secondary_model.feature_selection.main.build_secondary_greedy_filtered_features_csv", side_effect=greedy_filtered_csv),
            patch("core.src.secondary_model.feature_selection.main.run_feature_corr_pca", side_effect=fake_run_feature_corr_pca),
            patch("core.src.secondary_model.feature_selection.main.run_greedy_forward_selection", side_effect=fake_run_greedy_forward_selection),
        ):
            run_secondary_feature_selection(feature_path)

        assert resumed_target.name not in captured_corr_targets
        assert resumed_target.name in captured_greedy_targets
