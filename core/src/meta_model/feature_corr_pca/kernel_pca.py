from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas._libs.tslibs.nattype import NaTType

from core.src.meta_model.feature_corr_pca.config import (
    DEFAULT_COMPONENTS_PER_GROUP,
    DEFAULT_KERNEL,
    MAX_KERNEL_PCA_TRAIN_ROWS,
    TRAIN_END_DATE,
    resolve_max_workers,
)
from core.src.meta_model.feature_corr_pca.dataset_utils import select_train_frame_for_correlation
from core.src.meta_model.feature_corr_pca.models import KernelPCAGroupModel

LOGGER: logging.Logger = logging.getLogger(__name__)


def _normalize_timestamp(value: date | pd.Timestamp | None, *, label: str) -> pd.Timestamp | None:
    if value is None:
        return None
    normalized = pd.Timestamp(value)
    if isinstance(normalized, NaTType) or pd.isna(normalized):
        raise ValueError(f"{label} cannot be NaT.")
    return normalized


def rbf_kernel(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    left_sq = np.sum(left * left, axis=1)[:, None]
    right_sq = np.sum(right * right, axis=1)[None, :]
    squared_distance = np.maximum(left_sq + right_sq - (2.0 * left @ right.T), 0.0)
    return np.exp(-gamma * squared_distance)


def fit_kernel_pca_projection(
    train_values: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    kernel_fit = rbf_kernel(train_values, train_values, gamma)
    row_mean = kernel_fit.mean(axis=0)
    grand_mean = float(kernel_fit.mean())
    centered_kernel = kernel_fit - row_mean[None, :] - row_mean[:, None] + grand_mean
    eigenvalues, eigenvectors = np.linalg.eigh(centered_kernel)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    positive_indices = [
        index for index in sorted_indices if float(eigenvalues[index]) > 1e-12
    ]
    if not positive_indices:
        raise ValueError("Kernel PCA could not find a positive eigenvalue.")
    principal_index = positive_indices[0]
    principal_value = float(eigenvalues[principal_index])
    principal_vector = eigenvectors[:, principal_index] / np.sqrt(principal_value)
    return principal_vector, row_mean, grand_mean


def transform_kernel_pca_projection(
    full_values: np.ndarray,
    train_values: np.ndarray,
    gamma: float,
    principal_vector: np.ndarray,
    train_row_mean: np.ndarray,
    train_grand_mean: float,
) -> np.ndarray:
    kernel_new = rbf_kernel(full_values, train_values, gamma)
    new_row_mean = kernel_new.mean(axis=1, keepdims=True)
    centered_kernel = (
        kernel_new
        - train_row_mean[None, :]
        - new_row_mean
        + train_grand_mean
    )
    return centered_kernel @ principal_vector


def select_fit_rows(train_complete: pd.DataFrame) -> pd.DataFrame:
    if len(train_complete) <= MAX_KERNEL_PCA_TRAIN_ROWS:
        return train_complete.reset_index(drop=True)
    selected_positions = np.linspace(
        0,
        len(train_complete) - 1,
        num=MAX_KERNEL_PCA_TRAIN_ROWS,
        dtype=int,
    )
    unique_positions = np.unique(selected_positions)
    return train_complete.iloc[unique_positions].reset_index(drop=True)


def load_train_group_frame(
    feature_parquet_path: Path,
    group_columns: list[str],
    train_end_date: date | pd.Timestamp | None = None,
    start_date: date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    filter_end_date = _normalize_timestamp(train_end_date, label="train_end_date")
    if filter_end_date is None:
        fallback_end_date = pd.Timestamp(TRAIN_END_DATE)
        if isinstance(fallback_end_date, NaTType) or pd.isna(fallback_end_date):
            raise ValueError("Default TRAIN_END_DATE cannot be NaT.")
        filter_end_date = fallback_end_date
    filters: list[list[tuple[str, str, pd.Timestamp]]] = [[("date", "<=", filter_end_date)]]
    if start_date is not None:
        filter_start_date = _normalize_timestamp(start_date, label="start_date")
        if filter_start_date is None:
            raise ValueError("start_date normalization unexpectedly returned None.")
        filters = [[("date", ">=", filter_start_date), ("date", "<=", filter_end_date)]]
    return pd.read_parquet(
        feature_parquet_path,
        columns=group_columns,
        filters=filters,
    )


def fit_group_model_from_parquet(
    feature_parquet_path: Path,
    group_columns: list[str],
    group_index: int,
    kernel: str = DEFAULT_KERNEL,
    n_components: int = DEFAULT_COMPONENTS_PER_GROUP,
    train_end_date: date | pd.Timestamp | None = None,
    start_date: date | pd.Timestamp | None = None,
) -> tuple[KernelPCAGroupModel | None, dict[str, Any]]:
    component_feature_name = f"corr_pca_group_{group_index:03d}"
    mapping_row: dict[str, Any] = {
        "component_feature_name": component_feature_name,
        "member_features": group_columns,
        "group_size": len(group_columns),
        "kernel": kernel,
        "n_components": n_components,
    }
    if kernel != "rbf":
        mapping_row["status"] = "skipped_unsupported_kernel"
        return None, mapping_row

    train_frame = load_train_group_frame(
        feature_parquet_path,
        group_columns,
        train_end_date=train_end_date,
        start_date=start_date,
    )
    train_complete = train_frame.loc[:, group_columns].dropna().reset_index(drop=True)
    mapping_row["train_rows_used"] = int(len(train_complete))
    if len(train_complete) < 2:
        mapping_row["status"] = "skipped_insufficient_complete_train_rows"
        return None, mapping_row

    fit_frame = select_fit_rows(train_complete)
    train_complete_values = train_complete.to_numpy(dtype=np.float64, copy=False)
    fit_values = fit_frame.to_numpy(dtype=np.float64, copy=False)
    means = train_complete_values.mean(axis=0)
    stds = train_complete_values.std(axis=0)
    stds = np.where(stds == 0.0, 1.0, stds)
    train_scaled_fit = (fit_values - means) / stds
    gamma = 1.0 / max(1, len(group_columns))

    try:
        principal_vector, train_row_mean, train_grand_mean = fit_kernel_pca_projection(
            train_scaled_fit,
            gamma,
        )
    except ValueError:
        mapping_row["status"] = "skipped_non_invertible_train_kernel"
        return None, mapping_row

    mapping_row["status"] = "applied"
    mapping_row["fit_rows_used"] = int(len(fit_frame))
    model = KernelPCAGroupModel(
        component_feature_name=component_feature_name,
        member_features=tuple(group_columns),
        means=means,
        stds=stds,
        train_scaled_fit=train_scaled_fit,
        gamma=gamma,
        principal_vector=principal_vector,
        train_row_mean=train_row_mean,
        train_grand_mean=train_grand_mean,
        train_complete_rows=int(len(train_complete)),
        fit_rows_used=int(len(fit_frame)),
    )
    return model, mapping_row


def fit_group_models_from_parquet(
    feature_parquet_path: Path,
    correlated_groups: list[list[str]],
    kernel: str = DEFAULT_KERNEL,
    n_components_per_group: int = DEFAULT_COMPONENTS_PER_GROUP,
    train_end_date: date | pd.Timestamp | None = None,
    start_date: date | pd.Timestamp | None = None,
    max_workers: int | None = None,
) -> tuple[list[KernelPCAGroupModel], list[dict[str, Any]], list[dict[str, Any]]]:
    worker_count = min(resolve_max_workers(max_workers), max(1, len(correlated_groups)))
    if not correlated_groups:
        return [], [], []

    indexed_groups = list(enumerate(correlated_groups, start=1))
    total_groups = len(indexed_groups)
    stage_started_at = time.perf_counter()
    LOGGER.info(
        "Starting Kernel PCA group fitting stage: %d correlated groups | workers=%d.",
        total_groups,
        worker_count,
    )

    def _fit(indexed_group: tuple[int, list[str]]) -> tuple[int, KernelPCAGroupModel | None, dict[str, Any]]:
        group_index, group_columns = indexed_group
        model, mapping_row = fit_group_model_from_parquet(
            feature_parquet_path=feature_parquet_path,
            group_columns=group_columns,
            group_index=group_index,
            kernel=kernel,
            n_components=n_components_per_group,
            train_end_date=train_end_date,
            start_date=start_date,
        )
        return group_index, model, mapping_row

    if worker_count == 1:
        results = []
        for completed_count, indexed_group in enumerate(indexed_groups, start=1):
            results.append(_fit(indexed_group))
            LOGGER.info(
                "Kernel PCA fit progress: %d/%d groups processed | elapsed=%.2fs",
                completed_count,
                total_groups,
                time.perf_counter() - stage_started_at,
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_group = {
                executor.submit(_fit, indexed_group): indexed_group[0]
                for indexed_group in indexed_groups
            }
            results = []
            for completed_count, future in enumerate(as_completed(future_to_group), start=1):
                results.append(future.result())
                LOGGER.info(
                    "Kernel PCA fit progress: %d/%d groups processed | elapsed=%.2fs",
                    completed_count,
                    total_groups,
                    time.perf_counter() - stage_started_at,
                )

    results.sort(key=lambda item: item[0])
    models: list[KernelPCAGroupModel] = []
    applied_groups: list[dict[str, Any]] = []
    skipped_groups: list[dict[str, Any]] = []
    for _, model, mapping_row in results:
        if model is None:
            skipped_groups.append(mapping_row)
            continue
        models.append(model)
        applied_groups.append(mapping_row)

    LOGGER.info(
        "Kernel PCA fit stage completed: %d applied, %d skipped | workers=%d | elapsed=%.2fs.",
        len(applied_groups),
        len(skipped_groups),
        worker_count,
        time.perf_counter() - stage_started_at,
    )
    return models, applied_groups, skipped_groups


def transform_group_batch(
    batch_frame: pd.DataFrame,
    model: KernelPCAGroupModel,
) -> np.ndarray:
    group_columns = list(model.member_features)
    valid_mask = batch_frame.loc[:, group_columns].notna().all(axis=1).to_numpy()
    component_values = np.full(len(batch_frame), np.nan, dtype=np.float32)
    if not bool(valid_mask.any()):
        return component_values

    full_values = batch_frame.loc[valid_mask, group_columns].to_numpy(dtype=np.float64, copy=False)
    full_scaled = (full_values - model.means) / model.stds
    transformed = transform_kernel_pca_projection(
        full_scaled,
        model.train_scaled_fit,
        model.gamma,
        model.principal_vector,
        model.train_row_mean,
        model.train_grand_mean,
    )
    component_values[valid_mask] = transformed.astype(np.float32, copy=False)
    return component_values


def apply_feature_corr_kernel_pca(
    data: pd.DataFrame,
    correlation_threshold: float,
    kernel: str = DEFAULT_KERNEL,
    n_components_per_group: int = DEFAULT_COMPONENTS_PER_GROUP,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from core.src.meta_model.feature_corr_pca.correlation import find_correlated_feature_groups
    from core.src.meta_model.feature_corr_pca.dataset_utils import build_candidate_feature_columns

    feature_columns = build_candidate_feature_columns(data.columns)
    train_data = select_train_frame_for_correlation(data)
    correlated_groups = find_correlated_feature_groups(
        train_data,
        feature_columns=feature_columns,
        threshold=correlation_threshold,
    )

    transformed = data.copy()
    applied_groups: list[dict[str, Any]] = []
    skipped_groups: list[dict[str, Any]] = []
    columns_to_drop: set[str] = set()

    for group_index, group_columns in enumerate(correlated_groups, start=1):
        component_feature_name = f"corr_pca_group_{group_index:03d}"
        mapping_row: dict[str, Any] = {
            "component_feature_name": component_feature_name,
            "member_features": group_columns,
            "group_size": len(group_columns),
            "kernel": kernel,
            "n_components": n_components_per_group,
        }
        if kernel != "rbf":
            mapping_row["status"] = "skipped_unsupported_kernel"
            skipped_groups.append(mapping_row)
            continue

        train_complete = train_data.loc[:, group_columns].dropna().reset_index(drop=True)
        mapping_row["train_rows_used"] = int(len(train_complete))
        if len(train_complete) < 2:
            mapping_row["status"] = "skipped_insufficient_complete_train_rows"
            skipped_groups.append(mapping_row)
            continue

        fit_frame = select_fit_rows(train_complete)
        train_complete_values = train_complete.to_numpy(dtype=np.float64, copy=False)
        fit_values = fit_frame.to_numpy(dtype=np.float64, copy=False)
        means = train_complete_values.mean(axis=0)
        stds = train_complete_values.std(axis=0)
        stds = np.where(stds == 0.0, 1.0, stds)
        train_scaled_fit = (fit_values - means) / stds
        gamma = 1.0 / max(1, len(group_columns))

        try:
            principal_vector, train_row_mean, train_grand_mean = fit_kernel_pca_projection(
                train_scaled_fit,
                gamma,
            )
        except ValueError:
            mapping_row["status"] = "skipped_non_invertible_train_kernel"
            skipped_groups.append(mapping_row)
            continue

        valid_mask = transformed.loc[:, group_columns].notna().all(axis=1)
        component_values = np.full(len(transformed), np.nan, dtype=np.float64)
        if bool(valid_mask.any()):
            full_values = transformed.loc[valid_mask, group_columns].to_numpy(dtype=np.float64, copy=False)
            full_scaled = (full_values - means) / stds
            component_values[valid_mask.to_numpy()] = transform_kernel_pca_projection(
                full_scaled,
                train_scaled_fit,
                gamma,
                principal_vector,
                train_row_mean,
                train_grand_mean,
            )

        transformed[component_feature_name] = component_values
        columns_to_drop.update(group_columns)
        mapping_row["status"] = "applied"
        mapping_row["fit_rows_used"] = int(len(fit_frame))
        applied_groups.append(mapping_row)

    if columns_to_drop:
        transformed = transformed.drop(columns=sorted(columns_to_drop))

    mapping: dict[str, Any] = {
        "correlation_threshold": correlation_threshold,
        "kernel": kernel,
        "n_components_per_group": n_components_per_group,
        "train_end_date": str(TRAIN_END_DATE),
        "applied_groups": applied_groups,
        "skipped_groups": skipped_groups,
        "retained_original_feature_count": len(
            [column for column in transformed.columns if column not in ("date", "ticker", "stock_close_price", "target_main", "dataset_split", "row_position")]
        ),
    }
    return transformed, mapping
