from __future__ import annotations

import gc
import importlib
import importlib.util
import logging
from dataclasses import dataclass
from typing import Any

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
    features: Any,
    labels: Any,
    feature_names: list[str],
    *,
    prefer_quantile: bool = True,
) -> Any:
    if prefer_quantile:
        quantile_dmatrix = getattr(xgb_module, "QuantileDMatrix", None)
        if callable(quantile_dmatrix):
            try:
                return quantile_dmatrix(features, label=labels, feature_names=feature_names)
            except Exception:
                pass
    return xgb_module.DMatrix(features, label=labels, feature_names=feature_names)


def _load_cupy_module() -> Any | None:
    cupy_spec = importlib.util.find_spec("cupy")
    if cupy_spec is None:
        return None
    try:
        return importlib.import_module("cupy")
    except Exception:
        return None


def _build_fold_matrix_cache_gpu_resident(
    dataset_bundle: OptimizationDatasetBundle,
    fold_contexts: list[FoldEvaluationContext],
    *,
    xgb_module: Any,
    cupy_module: Any,
    gpu_device_id: int,
) -> dict[int, CachedFoldMatrixBundle]:
    with cupy_module.cuda.Device(gpu_device_id):
        gpu_feature_matrix = cupy_module.asarray(dataset_bundle.feature_matrix, dtype=cupy_module.float32)
        gpu_target_array = cupy_module.asarray(dataset_bundle.target_array, dtype=cupy_module.float32)

        cache_by_fold_index: dict[int, CachedFoldMatrixBundle] = {}
        for fold_context in fold_contexts:
            fold = fold_context.fold
            validation_indices = cupy_module.asarray(fold.validation_indices, dtype=cupy_module.int64)
            validation_features = cupy_module.take(gpu_feature_matrix, validation_indices, axis=0)
            validation_labels = cupy_module.take(gpu_target_array, validation_indices, axis=0)
            validation_matrix = _build_dmatrix(
                xgb_module,
                validation_features,
                validation_labels,
                dataset_bundle.feature_columns,
                prefer_quantile=False,
            )
            del validation_indices
            del validation_features
            del validation_labels

            cached_train_windows: list[CachedTrainWindowMatrix] = []
            for train_window in fold_context.train_windows:
                train_indices = cupy_module.asarray(train_window.train_indices, dtype=cupy_module.int64)
                train_features = cupy_module.take(gpu_feature_matrix, train_indices, axis=0)
                train_labels = cupy_module.take(gpu_target_array, train_indices, axis=0)
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
                del train_indices
                del train_features
                del train_labels

            cache_by_fold_index[fold.index] = CachedFoldMatrixBundle(
                fold_context=fold_context,
                validation_matrix=validation_matrix,
                train_windows=cached_train_windows,
            )

        del gpu_feature_matrix
        del gpu_target_array
        return cache_by_fold_index


def build_fold_matrix_cache(
    dataset_bundle: OptimizationDatasetBundle,
    fold_contexts: list[FoldEvaluationContext],
    *,
    xgb_module: Any,
    enabled: bool,
    gpu_device_id: int = 0,
) -> dict[int, CachedFoldMatrixBundle] | None:
    if not enabled:
        return None

    cupy_module = _load_cupy_module()
    if cupy_module is None:
        LOGGER.warning(
            "CUDA fold matrix cache disabled: CuPy unavailable. Continuing without persistent fold cache to avoid host RAM amplification.",
        )
        return None

    try:
        cache_by_fold_index = _build_fold_matrix_cache_gpu_resident(
            dataset_bundle,
            fold_contexts,
            xgb_module=xgb_module,
            cupy_module=cupy_module,
            gpu_device_id=gpu_device_id,
        )
    except Exception as error:
        LOGGER.warning(
            "CUDA fold matrix cache disabled (%s). Continuing without persistent fold cache to avoid host RAM amplification.",
            error,
        )
        return None

    gc.collect()
    LOGGER.info(
        "Built fold matrix cache for CUDA: backend=gpu | folds=%d | cached_windows=%d",
        len(cache_by_fold_index),
        sum(len(bundle.train_windows) for bundle in cache_by_fold_index.values()),
    )
    return cache_by_fold_index
