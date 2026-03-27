from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.data_preprocessing.main import (
    TARGET_COLUMN,
    filter_from_start_date,
    remove_rows_with_missing_values,
)
from core.src.meta_model.data.paths import FEATURES_OUTPUT_PARQUET
from core.src.meta_model.feature_corr_pca.main import run_feature_corr_pca
from core.src.meta_model.feature_selection_lag.greedy_forward_selection.main import (
    run_greedy_forward_selection,
)
from core.src.secondary_model.data.data_preprocessing.main import assign_secondary_dataset_splits
from core.src.secondary_model.data.paths import (
    SECONDARY_FEATURE_SELECTION_DIR,
    build_secondary_feature_selection_target_dir,
    build_secondary_feature_corr_pca_mapping_json,
    build_secondary_feature_corr_pca_output_parquet,
    build_secondary_feature_corr_pca_output_sample_csv,
    build_secondary_greedy_filtered_features_csv,
    build_secondary_greedy_filtered_features_parquet,
    build_secondary_greedy_scores_csv,
    build_secondary_greedy_scores_parquet,
    build_secondary_greedy_selected_features_csv,
    build_secondary_greedy_selected_features_parquet,
)
from core.src.secondary_model.data.targets import SECONDARY_TARGET_SPECS, SecondaryTargetSpec

LOGGER: logging.Logger = logging.getLogger(__name__)


def _feature_corr_pca_outputs_exist(target_spec: SecondaryTargetSpec) -> bool:
    expected_paths = (
        build_secondary_feature_corr_pca_output_parquet(target_spec.name),
        build_secondary_feature_corr_pca_output_sample_csv(target_spec.name),
        build_secondary_feature_corr_pca_mapping_json(target_spec.name),
    )
    return all(path.exists() for path in expected_paths)


def _greedy_outputs_exist(target_spec: SecondaryTargetSpec) -> bool:
    expected_paths = (
        build_secondary_greedy_scores_parquet(target_spec.name),
        build_secondary_greedy_scores_csv(target_spec.name),
        build_secondary_greedy_selected_features_parquet(target_spec.name),
        build_secondary_greedy_selected_features_csv(target_spec.name),
        build_secondary_greedy_filtered_features_parquet(target_spec.name),
        build_secondary_greedy_filtered_features_csv(target_spec.name),
    )
    return all(path.exists() for path in expected_paths)


def build_secondary_selection_scaffold(
    feature_parquet_path: Path,
    target_spec: SecondaryTargetSpec,
) -> pd.DataFrame:
    required_columns = ["date", "ticker", *target_spec.required_metadata_columns]
    source = pd.read_parquet(feature_parquet_path, columns=required_columns)
    ordered = pd.DataFrame(source.sort_values(["date", "ticker"]).reset_index(drop=True))
    ordered["row_position"] = np.arange(len(ordered), dtype=np.int64)
    filtered = filter_from_start_date(ordered)
    targeted = target_spec.build_target(filtered)
    split_ready = assign_secondary_dataset_splits(targeted)
    cleaned = remove_rows_with_missing_values(split_ready, required_columns=[TARGET_COLUMN])
    scaffold = pd.DataFrame(
        cleaned.loc[:, ["row_position", "date", "ticker", TARGET_COLUMN, "dataset_split"]]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True),
    )
    LOGGER.info(
        "Built secondary selection scaffold for %s: %d rows (%d train / %d val / %d test).",
        target_spec.name,
        len(scaffold),
        int((scaffold["dataset_split"] == "train").sum()),
        int((scaffold["dataset_split"] == "val").sum()),
        int((scaffold["dataset_split"] == "test").sum()),
    )
    return scaffold


def derive_secondary_train_end_date(
    feature_parquet_path: Path,
    target_spec: SecondaryTargetSpec,
) -> pd.Timestamp:
    scaffold = build_secondary_selection_scaffold(feature_parquet_path, target_spec)
    train_dates = pd.to_datetime(scaffold.loc[scaffold["dataset_split"] == "train", "date"])
    if train_dates.empty:
        raise ValueError(f"Secondary selection scaffold for {target_spec.name} does not contain any train rows.")
    train_end_date_raw = train_dates.max()
    if isinstance(train_end_date_raw, type(pd.NaT)) or pd.isna(train_end_date_raw):
        raise ValueError(f"Secondary train end date for {target_spec.name} cannot be NaT.")
    return train_end_date_raw


def run_secondary_feature_selection(
    feature_parquet_path: Path = FEATURES_OUTPUT_PARQUET,
) -> dict[str, Path]:
    SECONDARY_FEATURE_SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}

    for target_spec in SECONDARY_TARGET_SPECS:
        target_output_dir = build_secondary_feature_selection_target_dir(target_spec.name)
        target_output_dir.mkdir(parents=True, exist_ok=True)
        corr_output_path = build_secondary_feature_corr_pca_output_parquet(target_spec.name)
        filtered_output_path = build_secondary_greedy_filtered_features_parquet(target_spec.name)

        if _greedy_outputs_exist(target_spec):
            LOGGER.info(
                "Skipping secondary feature selection for %s because greedy outputs already exist.",
                target_spec.name,
            )
            output_paths[target_spec.name] = filtered_output_path
            continue

        selection_scaffold = build_secondary_selection_scaffold(feature_parquet_path, target_spec)
        LOGGER.info(
            (
                "Secondary feature selection started for %s: input=%s | "
                "corr_pca_ready=%s"
            ),
            target_spec.name,
            feature_parquet_path,
            _feature_corr_pca_outputs_exist(target_spec),
        )
        if not _feature_corr_pca_outputs_exist(target_spec):
            secondary_train_end_date = derive_secondary_train_end_date(feature_parquet_path, target_spec)
            run_feature_corr_pca(
                feature_parquet_path=feature_parquet_path,
                output_parquet_path=corr_output_path,
                output_sample_csv_path=build_secondary_feature_corr_pca_output_sample_csv(target_spec.name),
                output_mapping_json_path=build_secondary_feature_corr_pca_mapping_json(target_spec.name),
                train_end_date=secondary_train_end_date,
                return_transformed_data=False,
            )
        else:
            LOGGER.info(
                "Skipping corr+pca for %s because its outputs already exist.",
                target_spec.name,
            )
        run_greedy_forward_selection(
            feature_parquet_path=corr_output_path,
            selection_scaffold=selection_scaffold,
            scores_parquet_path=build_secondary_greedy_scores_parquet(target_spec.name),
            scores_csv_path=build_secondary_greedy_scores_csv(target_spec.name),
            selected_features_parquet_path=build_secondary_greedy_selected_features_parquet(target_spec.name),
            selected_features_csv_path=build_secondary_greedy_selected_features_csv(target_spec.name),
            filtered_features_parquet_path=filtered_output_path,
            filtered_features_csv_path=build_secondary_greedy_filtered_features_csv(target_spec.name),
            metadata_columns_to_keep=("date", "ticker", *target_spec.required_metadata_columns),
        )
        output_paths[target_spec.name] = filtered_output_path
        LOGGER.info(
            "Secondary feature selection completed for %s: %s",
            target_spec.name,
            filtered_output_path,
        )

    return output_paths


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_secondary_feature_selection()
    LOGGER.info("Secondary feature selection pipeline completed.")


__all__ = [
    "build_secondary_selection_scaffold",
    "derive_secondary_train_end_date",
    "main",
    "run_secondary_feature_selection",
]


if __name__ == "__main__":
    main()
