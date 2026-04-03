from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from core.src.meta_model.data.data_preprocessing.main import save_preprocessed_dataset
from core.src.meta_model.data.paths import (
    DATA_FEATURE_SELECTION_DIR,
    PREPROCESSED_OUTPUT_PARQUET,
)
from core.src.meta_model.data.registry import (
    build_feature_registry,
    save_feature_registry,
    save_feature_schema_manifest,
)
from core.src.meta_model.model_contract import (
    DATE_COLUMN,
    MODEL_TARGET_COLUMN,
    SPLIT_COLUMN,
    TICKER_COLUMN,
    TRAIN_SPLIT_NAME,
    is_excluded_feature_column,
)

LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSelectionMetadata:
    frame: pd.DataFrame
    target_values: np.ndarray
    ordered_dates: np.ndarray
    canonical_order: np.ndarray
    train_row_indices: np.ndarray
    available_columns: tuple[str, ...]

    @property
    def row_count(self) -> int:
        return len(self.frame)


@dataclass(frozen=True)
class FeatureSelectionInputInventory:
    row_groups: int
    total_columns: int
    excluded_columns: int
    numeric_feature_columns: int
    non_numeric_non_excluded_columns: int


@dataclass(frozen=True)
class ParquetSchemaInfo:
    field_names: tuple[str, ...]
    fields: tuple[pa.Field, ...]
    row_group_count: int


@dataclass(frozen=True)
class FeatureSelectionOutputBundle:
    output_dir: Path
    sfi_scores_parquet: Path
    sfi_scores_csv: Path
    stability_scores_parquet: Path
    stability_scores_csv: Path
    selected_features_parquet: Path
    selected_features_csv: Path
    filtered_dataset_parquet: Path
    filtered_dataset_csv: Path
    feature_registry_parquet: Path
    feature_registry_json: Path
    input_inventory_json: Path
    schema_manifest_json: Path
    group_manifest_parquet: Path
    group_manifest_csv: Path
    linear_pruning_audit_parquet: Path
    linear_pruning_audit_csv: Path
    distance_correlation_audit_parquet: Path
    distance_correlation_audit_csv: Path
    mda_group_scores_parquet: Path
    mda_group_scores_csv: Path
    mda_final_scores_parquet: Path
    mda_final_scores_csv: Path
    wrapper_search_parquet: Path
    wrapper_search_csv: Path
    summary_json: Path


def _is_numeric_schema_field(field: pa.Field) -> bool:
    pandas_dtype: object = field.type.to_pandas_dtype()
    return bool(pd.api.types.is_numeric_dtype(pandas_dtype))


def _schema_field_names(schema: pa.Schema) -> tuple[str, ...]:
    return tuple(str(field_name) for field_name in cast(list[object], schema.names))


def _schema_fields(schema: pa.Schema, field_names: tuple[str, ...]) -> tuple[pa.Field, ...]:
    return tuple(cast(pa.Field, schema.field(field_name)) for field_name in field_names)


def _load_parquet_schema(dataset_path: Path) -> ParquetSchemaInfo:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found for feature selection: {dataset_path}",
        )
    parquet_file = pq.ParquetFile(dataset_path)
    schema = cast(pa.Schema, parquet_file.schema_arrow)
    field_names = _schema_field_names(schema)
    return ParquetSchemaInfo(
        field_names=field_names,
        fields=_schema_fields(schema, field_names),
        row_group_count=int(parquet_file.num_row_groups),
    )


def build_feature_selection_input_inventory(
    dataset_path: Path = PREPROCESSED_OUTPUT_PARQUET,
) -> FeatureSelectionInputInventory:
    schema_info = _load_parquet_schema(dataset_path)
    excluded_columns = 0
    numeric_feature_columns = 0
    non_numeric_non_excluded_columns = 0
    for field in schema_info.fields:
        if is_excluded_feature_column(field.name):
            excluded_columns += 1
            continue
        if _is_numeric_schema_field(field):
            numeric_feature_columns += 1
            continue
        non_numeric_non_excluded_columns += 1
    return FeatureSelectionInputInventory(
        row_groups=schema_info.row_group_count,
        total_columns=len(schema_info.field_names),
        excluded_columns=excluded_columns,
        numeric_feature_columns=numeric_feature_columns,
        non_numeric_non_excluded_columns=non_numeric_non_excluded_columns,
    )


def save_feature_selection_input_inventory(
    inventory: FeatureSelectionInputInventory,
    path: Path | None = None,
) -> None:
    resolved_path = path or build_default_feature_selection_output_bundle().input_inventory_json
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(inventory), handle, indent=2, sort_keys=True)
    LOGGER.info("Saved feature-selection input inventory: %s", resolved_path)


def build_feature_selection_output_bundle(
    output_dir: Path,
    *,
    filtered_dataset_parquet_name: str,
    filtered_dataset_csv_name: str,
) -> FeatureSelectionOutputBundle:
    return FeatureSelectionOutputBundle(
        output_dir=output_dir,
        sfi_scores_parquet=output_dir / "feature_sfi_scores.parquet",
        sfi_scores_csv=output_dir / "feature_sfi_scores.csv",
        stability_scores_parquet=output_dir / "feature_stability_scores.parquet",
        stability_scores_csv=output_dir / "feature_stability_scores.csv",
        selected_features_parquet=output_dir / "feature_stability_selected.parquet",
        selected_features_csv=output_dir / "feature_stability_selected.csv",
        filtered_dataset_parquet=output_dir / filtered_dataset_parquet_name,
        filtered_dataset_csv=output_dir / filtered_dataset_csv_name,
        feature_registry_parquet=output_dir / "feature_registry.parquet",
        feature_registry_json=output_dir / "feature_registry.json",
        input_inventory_json=output_dir / "input_inventory.json",
        schema_manifest_json=output_dir / "feature_schema_manifest.json",
        group_manifest_parquet=output_dir / "feature_group_manifest.parquet",
        group_manifest_csv=output_dir / "feature_group_manifest.csv",
        linear_pruning_audit_parquet=output_dir / "feature_linear_pruning_audit.parquet",
        linear_pruning_audit_csv=output_dir / "feature_linear_pruning_audit.csv",
        distance_correlation_audit_parquet=output_dir / "feature_distance_correlation_audit.parquet",
        distance_correlation_audit_csv=output_dir / "feature_distance_correlation_audit.csv",
        mda_group_scores_parquet=output_dir / "feature_mda_group_scores.parquet",
        mda_group_scores_csv=output_dir / "feature_mda_group_scores.csv",
        mda_final_scores_parquet=output_dir / "feature_mda_final_scores.parquet",
        mda_final_scores_csv=output_dir / "feature_mda_final_scores.csv",
        wrapper_search_parquet=output_dir / "feature_wrapper_search.parquet",
        wrapper_search_csv=output_dir / "feature_wrapper_search.csv",
        summary_json=output_dir / "feature_selection_summary.json",
    )


def build_default_feature_selection_output_bundle() -> FeatureSelectionOutputBundle:
    return build_feature_selection_output_bundle(
        DATA_FEATURE_SELECTION_DIR,
        filtered_dataset_parquet_name="dataset_preprocessed_feature_selected.parquet",
        filtered_dataset_csv_name="dataset_preprocessed_feature_selected_sample_5pct.csv",
    )


def load_feature_selection_metadata(
    path: Path = PREPROCESSED_OUTPUT_PARQUET,
) -> FeatureSelectionMetadata:
    schema_info = _load_parquet_schema(path)
    data = pd.read_parquet(
        path,
        columns=[DATE_COLUMN, TICKER_COLUMN, SPLIT_COLUMN, MODEL_TARGET_COLUMN],
    )
    prepared = pd.DataFrame(data.copy())
    prepared[DATE_COLUMN] = pd.to_datetime(prepared[DATE_COLUMN])
    prepared[TICKER_COLUMN] = prepared[TICKER_COLUMN].astype("category")
    prepared[SPLIT_COLUMN] = prepared[SPLIT_COLUMN].astype("category")

    canonical_order = prepared.sort_values([DATE_COLUMN, TICKER_COLUMN]).index.to_numpy(
        dtype=np.int64,
        copy=False,
    )
    ordered = pd.DataFrame(prepared.take(canonical_order).reset_index(drop=True))
    frame = pd.DataFrame(ordered.loc[:, [DATE_COLUMN, TICKER_COLUMN, SPLIT_COLUMN]].copy())
    target_values = ordered[MODEL_TARGET_COLUMN].to_numpy(dtype=np.float64, copy=False)
    ordered_dates = frame[DATE_COLUMN].to_numpy()
    train_mask = frame[SPLIT_COLUMN].astype(str) == TRAIN_SPLIT_NAME
    train_row_indices = np.flatnonzero(train_mask.to_numpy())
    if train_row_indices.size == 0:
        raise ValueError("Feature selection dataset does not contain any train rows.")
    LOGGER.info(
        "Loaded feature-selection metadata: %d rows | train_rows=%d",
        len(frame),
        train_row_indices.size,
    )
    return FeatureSelectionMetadata(
        frame=frame,
        target_values=target_values,
        ordered_dates=ordered_dates,
        canonical_order=canonical_order,
        train_row_indices=train_row_indices,
        available_columns=schema_info.field_names,
    )


def load_preprocessed_feature_selection_dataset(
    path: Path = PREPROCESSED_OUTPUT_PARQUET,
) -> pd.DataFrame:
    data = pd.read_parquet(path)
    prepared = data.copy()
    prepared[DATE_COLUMN] = pd.to_datetime(prepared[DATE_COLUMN])
    return prepared.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)


def discover_selection_feature_columns(
    dataset_path: Path = PREPROCESSED_OUTPUT_PARQUET,
) -> list[str]:
    schema_info = _load_parquet_schema(dataset_path)
    feature_names = [
        str(field.name)
        for field in schema_info.fields
        if not is_excluded_feature_column(field.name) and _is_numeric_schema_field(field)
    ]
    return sorted(feature_names)


def discover_protected_columns(
    dataset_path: Path = PREPROCESSED_OUTPUT_PARQUET,
) -> list[str]:
    schema_info = _load_parquet_schema(dataset_path)
    return [
        str(field.name)
        for field in schema_info.fields
        if is_excluded_feature_column(field.name)
    ]


def iter_feature_batches(
    dataset_path: Path,
    feature_names: list[str],
    batch_size: int,
    canonical_order: np.ndarray,
) -> Iterator[tuple[list[str], pd.DataFrame]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be strictly positive.")
    for start in range(0, len(feature_names), batch_size):
        batch_feature_names = feature_names[start:start + batch_size]
        batch = pd.read_parquet(dataset_path, columns=batch_feature_names)
        ordered_batch = pd.DataFrame(batch.take(canonical_order))
        yield batch_feature_names, ordered_batch


def build_selected_feature_dataset(
    dataset_path: Path,
    selected_feature_names: list[str],
    retained_context_columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    protected_columns = discover_protected_columns(dataset_path)
    schema_info = _load_parquet_schema(dataset_path)
    available_columns = set(schema_info.field_names)
    retained_context_map = {
        source_column: retained_column
        for source_column, retained_column in (retained_context_columns or {}).items()
        if source_column in available_columns
    }
    selected_feature_name_set = set(selected_feature_names)
    non_overlapping_context_columns = [
        source_column
        for source_column in retained_context_map.keys()
        if source_column not in selected_feature_name_set
    ]
    ordered_columns = _deduplicate_preserving_order(
        [*protected_columns, *non_overlapping_context_columns, *selected_feature_names],
    )
    data = pd.read_parquet(dataset_path, columns=ordered_columns)
    prepared = pd.DataFrame(data.copy())
    prepared[DATE_COLUMN] = pd.to_datetime(prepared[DATE_COLUMN])
    if retained_context_map:
        overlap_columns = [
            source_column
            for source_column in retained_context_map
            if source_column in selected_feature_name_set
        ]
        if overlap_columns:
            LOGGER.info(
                "Feature selection filtered dataset context overlap: duplicating %d selected context features as hl_context_* copies",
                len(overlap_columns),
            )
        rename_map = {
            source_column: retained_column
            for source_column, retained_column in retained_context_map.items()
            if source_column not in selected_feature_name_set
        }
        if rename_map:
            prepared = prepared.rename(columns=rename_map)
        for source_column in overlap_columns:
            prepared[retained_context_map[source_column]] = prepared[source_column]
    return prepared.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)


def _deduplicate_preserving_order(column_names: list[str]) -> list[str]:
    unique_columns: list[str] = []
    seen: set[str] = set()
    for column_name in column_names:
        if column_name in seen:
            continue
        seen.add(column_name)
        unique_columns.append(column_name)
    return unique_columns


def save_feature_selection_outputs(
    score_frame: pd.DataFrame,
    selected_feature_names: list[str],
    filtered_dataset: pd.DataFrame,
    *,
    output_bundle: FeatureSelectionOutputBundle | None = None,
    input_inventory: FeatureSelectionInputInventory | None = None,
    group_manifest: pd.DataFrame | None = None,
    sfi_scores: pd.DataFrame | None = None,
    linear_pruning_audit: pd.DataFrame | None = None,
    distance_correlation_audit: pd.DataFrame | None = None,
    mda_group_scores: pd.DataFrame | None = None,
    mda_final_scores: pd.DataFrame | None = None,
    wrapper_search_history: pd.DataFrame | None = None,
    summary: dict[str, object] | None = None,
) -> None:
    bundle = output_bundle or build_default_feature_selection_output_bundle()
    bundle.output_dir.mkdir(parents=True, exist_ok=True)
    score_frame.to_parquet(bundle.stability_scores_parquet, index=False)
    score_frame.to_csv(bundle.stability_scores_csv, index=False)
    if sfi_scores is not None:
        sfi_scores.to_parquet(bundle.sfi_scores_parquet, index=False)
        sfi_scores.to_csv(bundle.sfi_scores_csv, index=False)

    selected_frame = pd.DataFrame({
        "feature_name": selected_feature_names,
        "selection_rank": np.arange(1, len(selected_feature_names) + 1, dtype=np.int64),
    })
    selected_frame.to_parquet(bundle.selected_features_parquet, index=False)
    selected_frame.to_csv(bundle.selected_features_csv, index=False)

    save_preprocessed_dataset(
        filtered_dataset,
        bundle.filtered_dataset_parquet,
        bundle.filtered_dataset_csv,
    )
    filtered_registry = build_feature_registry(filtered_dataset)
    save_feature_registry(
        filtered_registry,
        bundle.feature_registry_parquet,
        bundle.feature_registry_json,
    )
    save_feature_schema_manifest(
        selected_feature_names,
        bundle.schema_manifest_json,
    )
    if group_manifest is not None:
        group_manifest.to_parquet(bundle.group_manifest_parquet, index=False)
        group_manifest.to_csv(bundle.group_manifest_csv, index=False)
    if linear_pruning_audit is not None:
        linear_pruning_audit.to_parquet(bundle.linear_pruning_audit_parquet, index=False)
        linear_pruning_audit.to_csv(bundle.linear_pruning_audit_csv, index=False)
    if distance_correlation_audit is not None:
        distance_correlation_audit.to_parquet(
            bundle.distance_correlation_audit_parquet,
            index=False,
        )
        distance_correlation_audit.to_csv(bundle.distance_correlation_audit_csv, index=False)
    if mda_group_scores is not None:
        mda_group_scores.to_parquet(bundle.mda_group_scores_parquet, index=False)
        mda_group_scores.to_csv(bundle.mda_group_scores_csv, index=False)
    if mda_final_scores is not None:
        mda_final_scores.to_parquet(bundle.mda_final_scores_parquet, index=False)
        mda_final_scores.to_csv(bundle.mda_final_scores_csv, index=False)
    if wrapper_search_history is not None:
        wrapper_search_history.to_parquet(bundle.wrapper_search_parquet, index=False)
        wrapper_search_history.to_csv(bundle.wrapper_search_csv, index=False)
    if summary is not None:
        with bundle.summary_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    if input_inventory is not None:
        save_feature_selection_input_inventory(
            input_inventory,
            path=bundle.input_inventory_json,
        )
    LOGGER.info(
        "Saved feature-selection outputs: selected=%d | scores=%d",
        len(selected_feature_names),
        len(score_frame),
    )


__all__ = [
    "FeatureSelectionInputInventory",
    "FeatureSelectionMetadata",
    "FeatureSelectionOutputBundle",
    "build_feature_selection_output_bundle",
    "build_feature_selection_input_inventory",
    "build_default_feature_selection_output_bundle",
    "build_selected_feature_dataset",
    "discover_protected_columns",
    "discover_selection_feature_columns",
    "iter_feature_batches",
    "load_feature_selection_metadata",
    "load_preprocessed_feature_selection_dataset",
    "save_feature_selection_input_inventory",
    "save_feature_selection_outputs",
]
