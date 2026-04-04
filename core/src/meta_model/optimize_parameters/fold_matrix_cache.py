from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.src.meta_model.optimize_parameters.dataset import OptimizationDatasetBundle
from core.src.meta_model.optimize_parameters.fold_context import FoldEvaluationContext

LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedTrainWindowMatrix:
    label: str
    coverage_fraction: float
    train_matrix: Any


@dataclass(frozen=True)
class CachedFoldMatrixBundle:
    fold_context: FoldEvaluationContext
    validation_matrix: Any
    train_windows: list[CachedTrainWindowMatrix]


def _build_dmatrix(
    xgb_module: Any,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
) -> Any:
    quantile_dmatrix = getattr(xgb_module, "QuantileDMatrix", None)
    if callable(quantile_dmatrix):
        try:
            return quantile_dmatrix(features, label=labels, feature_names=feature_names)
        except Exception:
            pass
    return xgb_module.DMatrix(features, label=labels, feature_names=feature_names)


def build_fold_matrix_cache(
    dataset_bundle: OptimizationDatasetBundle,
    fold_contexts: list[FoldEvaluationContext],
    *,
    xgb_module: Any,
    enabled: bool,
) -> dict[int, CachedFoldMatrixBundle] | None:
    if not enabled:
        return None

    cache_by_fold_index: dict[int, CachedFoldMatrixBundle] = {}
    for fold_context in fold_contexts:
        fold = fold_context.fold
        validation_features = np.ascontiguousarray(
            dataset_bundle.feature_matrix[fold.validation_indices],
        )
        validation_labels = np.ascontiguousarray(
            dataset_bundle.target_array[fold.validation_indices],
        )
        validation_matrix = _build_dmatrix(
            xgb_module,
            validation_features,
            validation_labels,
            dataset_bundle.feature_columns,
        )
        del validation_features
        del validation_labels

        cached_train_windows: list[CachedTrainWindowMatrix] = []
        for train_window in fold_context.train_windows:
            train_features = np.ascontiguousarray(
                dataset_bundle.feature_matrix[train_window.train_indices],
            )
            train_labels = np.ascontiguousarray(
                dataset_bundle.target_array[train_window.train_indices],
            )
            train_matrix = _build_dmatrix(
                xgb_module,
                train_features,
                train_labels,
                dataset_bundle.feature_columns,
            )
            cached_train_windows.append(
                CachedTrainWindowMatrix(
                    label=train_window.label,
                    coverage_fraction=train_window.coverage_fraction,
                    train_matrix=train_matrix,
                ),
            )
            del train_features
            del train_labels

        cache_by_fold_index[fold.index] = CachedFoldMatrixBundle(
            fold_context=fold_context,
            validation_matrix=validation_matrix,
            train_windows=cached_train_windows,
        )

    gc.collect()
    LOGGER.info(
        "Built fold matrix cache for CUDA: folds=%d | cached_windows=%d",
        len(cache_by_fold_index),
        sum(len(bundle.train_windows) for bundle in cache_by_fold_index.values()),
    )
    return cache_by_fold_index
