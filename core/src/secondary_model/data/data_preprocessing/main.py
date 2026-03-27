from __future__ import annotations

import math
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.data_preprocessing.main import (
    SPLIT_COLUMN,
    TARGET_COLUMN,
    assign_dataset_splits as assign_meta_dataset_splits,
    drop_columns_with_missing_values,
    filter_from_start_date,
    forward_fill_features_by_ticker,
    load_feature_dataset,
    prune_correlated_features,
    remove_rows_with_missing_values,
    save_preprocessed_dataset,
    validate_no_missing_values,
)
from core.src.secondary_model.data.paths import (
    SECONDARY_DATA_PREPROCESSING_DIR,
    build_secondary_greedy_filtered_features_parquet,
)
from core.src.secondary_model.data.targets import (
    SECONDARY_TARGET_SPECS,
    create_future_drawdown_target,
    create_future_realized_vol_target,
    create_future_trend_target,
    create_future_volume_regime_target,
)

LOGGER: logging.Logger = logging.getLogger(__name__)

SECONDARY_TRAIN_FRACTION_OF_META_TRAIN: float = 0.2
SECONDARY_VAL_FRACTION_OF_META_TRAIN: float = 0.1


def assign_secondary_dataset_splits(
    data: pd.DataFrame,
    train_fraction_of_meta_train: float = SECONDARY_TRAIN_FRACTION_OF_META_TRAIN,
    val_fraction_of_meta_train: float = SECONDARY_VAL_FRACTION_OF_META_TRAIN,
) -> pd.DataFrame:
    if not 0.0 < train_fraction_of_meta_train <= 1.0:
        raise ValueError("train_fraction_of_meta_train must be in the interval (0, 1].")
    if not 0.0 <= val_fraction_of_meta_train <= 1.0:
        raise ValueError("val_fraction_of_meta_train must be in the interval [0, 1].")

    meta_split_ready = assign_meta_dataset_splits(data)
    train_dates = pd.Index(
        pd.to_datetime(
            meta_split_ready.loc[meta_split_ready[SPLIT_COLUMN] == "train", "date"],
        ).drop_duplicates().sort_values(),
    )
    if train_dates.empty:
        raise ValueError("Meta-model train split is empty; cannot derive secondary splits.")

    secondary_train_date_count = max(1, int(math.ceil(len(train_dates) * train_fraction_of_meta_train)))
    remaining_train_dates = max(0, len(train_dates) - secondary_train_date_count)
    secondary_val_date_count = min(
        remaining_train_dates,
        int(math.ceil(len(train_dates) * val_fraction_of_meta_train)),
    )
    secondary_train_dates = set(train_dates[:secondary_train_date_count].tolist())
    secondary_val_dates = set(
        train_dates[secondary_train_date_count : secondary_train_date_count + secondary_val_date_count].tolist(),
    )

    split_ready = meta_split_ready.copy()
    split_ready_dates = pd.to_datetime(split_ready["date"])
    original_meta_train_mask = split_ready[SPLIT_COLUMN] == "train"
    secondary_train_mask = original_meta_train_mask & split_ready_dates.isin(secondary_train_dates)
    secondary_val_mask = original_meta_train_mask & split_ready_dates.isin(secondary_val_dates)

    split_ready.loc[:, SPLIT_COLUMN] = "test"
    split_ready.loc[secondary_train_mask, SPLIT_COLUMN] = "train"
    split_ready.loc[secondary_val_mask, SPLIT_COLUMN] = "val"

    LOGGER.info(
        "Assigned secondary dataset splits: train=%d, val=%d, test=%d",
        int((split_ready[SPLIT_COLUMN] == "train").sum()),
        int((split_ready[SPLIT_COLUMN] == "val").sum()),
        int((split_ready[SPLIT_COLUMN] == "test").sum()),
    )
    return split_ready.sort_values(["date", "ticker"]).reset_index(drop=True)


def _save_split_datasets(data: pd.DataFrame, target_dir: Path) -> None:
    split_outputs: tuple[tuple[str, str, str], ...] = (
        ("train", "dataset_preprocessed_train.parquet", "dataset_preprocessed_train_sample_5pct.csv"),
        ("val", "dataset_preprocessed_val.parquet", "dataset_preprocessed_val_sample_5pct.csv"),
        ("test", "dataset_preprocessed_test.parquet", "dataset_preprocessed_test_sample_5pct.csv"),
    )
    for split_name, parquet_name, csv_name in split_outputs:
        split_df = pd.DataFrame(data.loc[data[SPLIT_COLUMN] == split_name].copy())
        save_preprocessed_dataset(split_df, target_dir / parquet_name, target_dir / csv_name)


def _build_target_output_dir(target_name: str) -> Path:
    return SECONDARY_DATA_PREPROCESSING_DIR / target_name


def run_secondary_target_preprocessing(
    feature_parquet_paths: dict[str, Path] | None = None,
) -> dict[str, pd.DataFrame]:
    SECONDARY_DATA_PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    output_datasets: dict[str, pd.DataFrame] = {}

    for target_spec in SECONDARY_TARGET_SPECS:
        LOGGER.info("Starting secondary preprocessing for target=%s", target_spec.name)
        source_path = (
            feature_parquet_paths[target_spec.name]
            if feature_parquet_paths is not None
            else build_secondary_greedy_filtered_features_parquet(target_spec.name)
        )
        featured = load_feature_dataset(source_path)
        filtered = filter_from_start_date(featured)
        targeted = target_spec.build_target(filtered)
        split_ready = assign_secondary_dataset_splits(targeted)
        forward_filled = forward_fill_features_by_ticker(
            split_ready,
            protected_columns=["date", "ticker", TARGET_COLUMN, SPLIT_COLUMN],
        )
        rows_ready = remove_rows_with_missing_values(forward_filled)
        columns_ready = drop_columns_with_missing_values(
            rows_ready,
            protected_columns=["date", "ticker", TARGET_COLUMN, SPLIT_COLUMN],
        )
        pruned = prune_correlated_features(columns_ready)
        validate_no_missing_values(pruned)

        target_dir = _build_target_output_dir(target_spec.name)
        save_preprocessed_dataset(
            pruned,
            target_dir / "dataset_preprocessed.parquet",
            target_dir / "dataset_preprocessed_sample_5pct.csv",
        )
        _save_split_datasets(pruned, target_dir)
        output_datasets[target_spec.name] = pruned
        LOGGER.info("Completed secondary preprocessing for target=%s", target_spec.name)

    return output_datasets


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_secondary_target_preprocessing()
    LOGGER.info("Secondary model data preprocessing pipeline completed.")


__all__ = [
    "SECONDARY_TARGET_SPECS",
    "SECONDARY_DATA_PREPROCESSING_DIR",
    "assign_secondary_dataset_splits",
    "create_future_drawdown_target",
    "create_future_realized_vol_target",
    "create_future_trend_target",
    "create_future_volume_regime_target",
    "main",
    "run_secondary_target_preprocessing",
]

if __name__ == "__main__":
    main()
