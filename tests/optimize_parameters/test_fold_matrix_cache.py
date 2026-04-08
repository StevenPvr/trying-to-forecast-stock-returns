from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.model_contract import MODEL_TARGET_COLUMN, STRUCTURAL_CATEGORICAL_FEATURE_COLUMNS
from core.src.meta_model.optimize_parameters.config import OptimizationConfig
from core.src.meta_model.optimize_parameters.cv import build_walk_forward_folds
from core.src.meta_model.optimize_parameters.dataset import (
    OptimizationDatasetBundle,
    build_optimization_dataset_bundle,
)
from core.src.meta_model.optimize_parameters.fold_context import build_fold_evaluation_contexts
from core.src.meta_model.optimize_parameters.fold_matrix_cache import (
    CachedFoldMatrixBundle,
    CachedTrainWindowMatrix,
    build_fold_matrix_cache,
)


class _FakeDMatrix:
    def __init__(self, data: Any, label: Any, feature_names: list[str] | None = None) -> None:
        self.data = data
        self.label = label
        self.feature_names = feature_names or []


class _FakeQuantileDMatrix(_FakeDMatrix):
    pass


class _FakeXGBoostModule:
    def __init__(self) -> None:
        setattr(self, "DMatrix", _FakeDMatrix)
        setattr(self, "QuantileDMatrix", _FakeQuantileDMatrix)


class _FakeCuPyDevice:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback


class _FakeCuPyCuda:
    @staticmethod
    def Device(device_id: int) -> _FakeCuPyDevice:
        del device_id
        return _FakeCuPyDevice()


class _FakeCuPyModule:
    float32 = np.float32
    int64 = np.int64
    cuda = _FakeCuPyCuda()

    @staticmethod
    def asarray(data: Any, dtype: Any = None) -> np.ndarray:
        return np.asarray(data, dtype=dtype)

    @staticmethod
    def take(array: Any, indices: Any, axis: int = 0) -> np.ndarray:
        return np.take(np.asarray(array), np.asarray(indices), axis=axis)


def _make_preprocessed_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, start, periods in (
        ("train", "2018-01-01", 20),
        ("val", "2019-02-01", 10),
    ):
        for idx, date in enumerate(pd.date_range(start, periods=periods, freq="B"), start=1):
            rows.append({
                "date": date,
                "ticker": "AAA",
                "target_main": 0.01 * idx,
                MODEL_TARGET_COLUMN: 0.01 * idx,
                "dataset_split": split_name,
                "feature_a": float(idx),
                "feature_b": float(idx * 2),
            })
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def _make_numeric_features_optimization_bundle() -> OptimizationDatasetBundle:
    data = _make_preprocessed_df()
    full = build_optimization_dataset_bundle(data, dataset_path=Path("synthetic.parquet"))
    numeric_columns = [
        name for name in full.feature_columns if name not in STRUCTURAL_CATEGORICAL_FEATURE_COLUMNS
    ]
    return OptimizationDatasetBundle(
        metadata=full.metadata,
        feature_columns=numeric_columns,
        feature_frame=pd.DataFrame(full.feature_frame.loc[:, numeric_columns].copy()),
        target_array=full.target_array,
    )


class TestFoldMatrixCache:
    def test_gpu_cache_uses_dmatrix_for_train_and_validation_to_keep_max_bin_dynamic(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bundle = _make_numeric_features_optimization_bundle()
        config = OptimizationConfig(fold_count=5, target_horizon_days=1)
        folds = build_walk_forward_folds(
            bundle.metadata,
            fold_count=config.fold_count,
            target_horizon_days=config.target_horizon_days,
        )
        fold_contexts = build_fold_evaluation_contexts(bundle, folds, config)

        monkeypatch.setattr(
            "core.src.meta_model.optimize_parameters.fold_matrix_cache._load_cupy_module",
            lambda: _FakeCuPyModule(),
        )

        cache = build_fold_matrix_cache(
            bundle,
            fold_contexts,
            xgb_module=_FakeXGBoostModule(),
            enabled=True,
        )

        assert cache is not None
        for cached_fold in cache.values():
            assert type(cached_fold.validation_matrix) is _FakeDMatrix
            for train_window in cached_fold.train_windows:
                assert type(train_window.train_matrix) is _FakeDMatrix

    def test_returns_none_when_cupy_unavailable_to_avoid_host_ram_cache(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        data = _make_preprocessed_df()
        bundle = build_optimization_dataset_bundle(data, dataset_path=Path("synthetic.parquet"))
        config = OptimizationConfig(fold_count=5, target_horizon_days=1)
        folds = build_walk_forward_folds(
            bundle.metadata,
            fold_count=config.fold_count,
            target_horizon_days=config.target_horizon_days,
        )
        fold_contexts = build_fold_evaluation_contexts(bundle, folds, config)

        monkeypatch.setattr(
            "core.src.meta_model.optimize_parameters.fold_matrix_cache._load_cupy_module",
            lambda: None,
        )

        cache = build_fold_matrix_cache(
            bundle,
            fold_contexts,
            xgb_module=_FakeXGBoostModule(),
            enabled=True,
        )

        assert cache is None

    def test_returns_gpu_cache_when_gpu_builder_succeeds(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bundle = _make_numeric_features_optimization_bundle()
        config = OptimizationConfig(fold_count=5, target_horizon_days=1)
        folds = build_walk_forward_folds(
            bundle.metadata,
            fold_count=config.fold_count,
            target_horizon_days=config.target_horizon_days,
        )
        fold_contexts = build_fold_evaluation_contexts(bundle, folds, config)
        sample_fold_context = fold_contexts[0]
        fake_cache = {
            sample_fold_context.fold.index: CachedFoldMatrixBundle(
                fold_context=sample_fold_context,
                validation_matrix=_FakeDMatrix([], []),
                train_windows=[
                    CachedTrainWindowMatrix(
                        label="full",
                        coverage_fraction=1.0,
                        train_matrix=_FakeQuantileDMatrix([], []),
                    ),
                ],
            ),
        }

        monkeypatch.setattr(
            "core.src.meta_model.optimize_parameters.fold_matrix_cache._load_cupy_module",
            lambda: object(),
        )

        def _gpu_builder(
            dataset_bundle: Any,
            fold_contexts: Any,
            *,
            xgb_module: Any,
            cupy_module: Any,
            gpu_device_id: int,
        ) -> dict[int, CachedFoldMatrixBundle]:
            del dataset_bundle, fold_contexts, xgb_module, cupy_module, gpu_device_id
            return fake_cache

        monkeypatch.setattr(
            "core.src.meta_model.optimize_parameters.fold_matrix_cache._build_fold_matrix_cache_gpu_resident",
            _gpu_builder,
        )

        cache = build_fold_matrix_cache(
            bundle,
            fold_contexts,
            xgb_module=_FakeXGBoostModule(),
            enabled=True,
        )

        assert cache is fake_cache

    def test_falls_back_to_validation_only_cache_when_full_gpu_cache_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bundle = _make_numeric_features_optimization_bundle()
        config = OptimizationConfig(fold_count=5, target_horizon_days=1)
        folds = build_walk_forward_folds(
            bundle.metadata,
            fold_count=config.fold_count,
            target_horizon_days=config.target_horizon_days,
        )
        fold_contexts = build_fold_evaluation_contexts(bundle, folds, config)
        sample_fold_context = fold_contexts[0]
        fake_validation_only_cache = {
            sample_fold_context.fold.index: CachedFoldMatrixBundle(
                fold_context=sample_fold_context,
                validation_matrix=_FakeDMatrix([], []),
                train_windows=[],
            ),
        }

        monkeypatch.setattr(
            "core.src.meta_model.optimize_parameters.fold_matrix_cache._load_cupy_module",
            lambda: object(),
        )

        def _raise_full_builder(*args: Any, **kwargs: Any) -> dict[int, CachedFoldMatrixBundle]:
            del args, kwargs
            raise RuntimeError("simulated oom")

        monkeypatch.setattr(
            "core.src.meta_model.optimize_parameters.fold_matrix_cache._build_fold_matrix_cache_gpu_resident",
            _raise_full_builder,
        )
        monkeypatch.setattr(
            "core.src.meta_model.optimize_parameters.fold_matrix_cache._build_validation_matrix_cache_gpu_resident",
            lambda dataset_bundle, fold_contexts, *, xgb_module, cupy_module, gpu_device_id: fake_validation_only_cache,
        )

        cache = build_fold_matrix_cache(
            bundle,
            fold_contexts,
            xgb_module=_FakeXGBoostModule(),
            enabled=True,
        )

        assert cache is fake_validation_only_cache

    def test_skips_gpu_cache_when_native_categorical_columns_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        data = _make_preprocessed_df()
        bundle = build_optimization_dataset_bundle(data, dataset_path=Path("synthetic.parquet"))
        config = OptimizationConfig(fold_count=5, target_horizon_days=1)
        folds = build_walk_forward_folds(
            bundle.metadata,
            fold_count=config.fold_count,
            target_horizon_days=config.target_horizon_days,
        )
        fold_contexts = build_fold_evaluation_contexts(bundle, folds, config)

        monkeypatch.setattr(
            "core.src.meta_model.optimize_parameters.fold_matrix_cache._load_cupy_module",
            lambda: _FakeCuPyModule(),
        )

        cache = build_fold_matrix_cache(
            bundle,
            fold_contexts,
            xgb_module=_FakeXGBoostModule(),
            enabled=True,
        )

        assert cache is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
