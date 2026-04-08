from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.paths import PREPROCESSED_OUTPUT_PARQUET
from core.src.meta_model.feature_selection.cache import FeatureSelectionRuntimeCache
from core.src.meta_model.feature_selection.config import FeatureSelectionConfig
from core.src.meta_model.feature_selection.cv import build_train_only_selection_folds
from core.src.meta_model.feature_selection.io import (
    FeatureSelectionOutputBundle,
    build_feature_selection_metadata_from_frame,
    build_feature_selection_input_inventory,
    build_selected_feature_dataset,
    build_default_feature_selection_output_bundle,
    discover_selection_feature_columns,
    load_sampled_train_feature_selection_dataset,
    load_feature_selection_metadata,
    materialize_feature_selection_dataset,
    read_parquet_column_names,
    save_feature_selection_outputs,
    subsample_train_feature_selection_metadata,
)
from core.src.meta_model.feature_selection.reporting import (
    validate_filtered_dataset_matches_selection,
)
from core.src.meta_model.feature_selection.selection_pipeline import run_robust_feature_selection
from core.src.meta_model.model_contract import (
    LABEL_EMBARGO_DAYS,
    is_excluded_feature_column,
    merge_structural_feature_names_into_selected,
)
from core.src.meta_model.model_contract import is_temporarily_disabled_alpha_feature_column

"""Feature selection orchestrator: load data, run SFI + pruning + search, save outputs."""

LOGGER: logging.Logger = logging.getLogger(__name__)

DEFAULT_RETAINED_CONTEXT_COLUMNS: dict[str, str] = {
    "stock_open_price": "hl_context_stock_open_price",
    "stock_high_price": "hl_context_stock_high_price",
    "stock_low_price": "hl_context_stock_low_price",
    "stock_close_price": "hl_context_stock_close_price",
    "stock_trading_volume": "hl_context_stock_trading_volume",
}


def build_selection_feature_columns(data: pd.DataFrame) -> list[str]:
    """Return sorted numeric feature columns eligible for selection."""
    raw_column_names = cast(list[object], data.columns.tolist())
    column_names = [str(column_name) for column_name in raw_column_names]
    return sorted(
        [
            column_name
            for column_name in column_names
            if not is_excluded_feature_column(column_name)
            and not is_temporarily_disabled_alpha_feature_column(column_name)
            and pd.api.types.is_numeric_dtype(data[column_name])
        ],
    )


def run_feature_selection(
    dataset_path: Path = PREPROCESSED_OUTPUT_PARQUET,
    config: FeatureSelectionConfig | None = None,
    *,
    output_bundle: FeatureSelectionOutputBundle | None = None,
    retained_context_columns: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full feature selection pipeline and return (score_frame, selected_frame, filtered_dataset)."""
    start_time = time.perf_counter()
    selection_config = config or FeatureSelectionConfig()
    resolved_output_bundle = output_bundle or build_default_feature_selection_output_bundle()
    resolved_retained_context_columns = (
        DEFAULT_RETAINED_CONTEXT_COLUMNS
        if retained_context_columns is None
        else retained_context_columns
    )
    LOGGER.info(
        "Feature selection started: dataset=%s | fold_count=%d | group_sample_size=%d | max_group_size=%d | parallel_workers=%d | beam_width=%d | pair_seed_group_limit=%d | max_active_matrix_gib=%.2f",
        dataset_path,
        selection_config.fold_count,
        selection_config.group_sample_size,
        selection_config.max_group_size,
        selection_config.parallel_workers,
        selection_config.search_beam_width,
        selection_config.pair_seed_group_limit,
        selection_config.max_active_matrix_gib,
    )
    input_inventory = build_feature_selection_input_inventory(dataset_path)
    LOGGER.info(
        "Feature selection inventory: row_groups=%d | total_columns=%d | numeric_feature_columns=%d | excluded_columns=%d | non_numeric_non_excluded_columns=%d",
        input_inventory.row_groups,
        input_inventory.total_columns,
        input_inventory.numeric_feature_columns,
        input_inventory.excluded_columns,
        input_inventory.non_numeric_non_excluded_columns,
    )
    metadata = load_feature_selection_metadata(dataset_path)
    sampled_metadata = subsample_train_feature_selection_metadata(
        metadata,
        train_sampling_fraction=selection_config.train_sampling_fraction,
        minimum_unique_dates=selection_config.fold_count + LABEL_EMBARGO_DAYS + 1,
    )
    feature_columns = discover_selection_feature_columns(dataset_path)
    if not feature_columns:
        raise ValueError("Feature selection found no numeric feature columns to score.")
    selection_dataset = load_sampled_train_feature_selection_dataset(
        dataset_path,
        sampled_metadata,
        feature_columns,
    )
    selection_metadata = build_feature_selection_metadata_from_frame(selection_dataset)
    LOGGER.info(
        "Feature selection candidates discovered: features=%d | train_rows=%d | available_columns=%d",
        len(feature_columns),
        selection_metadata.train_row_indices.size,
        len(selection_metadata.available_columns),
    )
    with materialize_feature_selection_dataset(selection_dataset) as selection_dataset_path:
        cache = FeatureSelectionRuntimeCache(
            selection_dataset_path,
            selection_metadata,
            random_seed=selection_config.random_seed,
            max_cache_gib=max(selection_config.max_active_matrix_gib / 2.0, 0.5),
        )
        folds = build_train_only_selection_folds(
            selection_metadata,
            selection_config.fold_count,
            label_embargo_days=LABEL_EMBARGO_DAYS,
        )
        LOGGER.info("Feature selection folds built: count=%d", len(folds))
        selection_result = run_robust_feature_selection(
            cache,
            folds,
            feature_columns,
            selection_config,
        )
    selected_feature_names = selection_result.selected_feature_names
    if not selected_feature_names:
        raise RuntimeError(_build_selection_failure_message(selection_result.score_frame))
    score_frame = selection_result.score_frame
    manifest_feature_names = merge_structural_feature_names_into_selected(
        selected_feature_names,
        available_columns=read_parquet_column_names(dataset_path),
    )
    filtered_dataset = build_selected_feature_dataset(
        dataset_path,
        manifest_feature_names,
        retained_context_columns=resolved_retained_context_columns,
    )
    validate_filtered_dataset_matches_selection(filtered_dataset, manifest_feature_names)
    save_feature_selection_outputs(
        score_frame,
        manifest_feature_names,
        filtered_dataset,
        output_bundle=resolved_output_bundle,
        input_inventory=input_inventory if selection_config.emit_input_inventory else None,
        group_manifest=selection_result.group_manifest,
        sfi_scores=selection_result.sfi_scores,
        linear_pruning_audit=selection_result.linear_pruning_audit,
        distance_correlation_audit=selection_result.distance_correlation_audit,
        target_correlation_audit=selection_result.target_correlation_audit,
        summary=selection_result.summary,
    )
    selected_frame = pd.DataFrame(
        {
            "feature_name": manifest_feature_names,
            "selection_rank": np.arange(1, len(manifest_feature_names) + 1, dtype=np.int64),
        },
    )
    LOGGER.info(
        "Feature selection completed: input_features=%d | selected_features=%d | dataset_rows=%d | elapsed=%.2fs",
        len(feature_columns),
        len(manifest_feature_names),
        len(filtered_dataset),
        time.perf_counter() - start_time,
    )
    return score_frame, selected_frame, filtered_dataset


def _build_selection_failure_message(score_frame: pd.DataFrame) -> str:
    if score_frame.empty:
        return "Feature selection did not retain any feature after SFI and pruning stages."
    sorted_history = score_frame.sort_values(
        ["objective_score", "daily_rank_ic_mean", "coverage_fraction", "feature_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    best_records = cast(list[dict[str, object]], sorted_history.to_dict(orient="records"))
    best_row = best_records[0]
    return (
        "Feature selection did not retain any feature after SFI and pruning stages. "
        f"Best candidate: feature={str(best_row['feature_name'])}, "
        f"sfi_objective={float(cast(float, best_row.get('objective_score', 0.0))):.6f}, "
        f"drop_reason={str(best_row.get('drop_reason', 'unknown'))}"
    )


def main() -> None:
    """Entry point for the feature selection pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_feature_selection()


__all__ = [
    "FeatureSelectionConfig",
    "DEFAULT_RETAINED_CONTEXT_COLUMNS",
    "build_selection_feature_columns",
    "run_feature_selection",
    "validate_filtered_dataset_matches_selection",
]


if __name__ == "__main__":
    main()
