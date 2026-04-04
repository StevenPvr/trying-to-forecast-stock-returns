from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.model_contract import MODEL_TARGET_COLUMN
from core.src.meta_model.optimize_parameters.config import OptimizationConfig
from core.src.meta_model.optimize_parameters.cv import build_walk_forward_folds
from core.src.meta_model.optimize_parameters.dataset import build_optimization_dataset_bundle
from core.src.meta_model.optimize_parameters.fold_context import build_fold_evaluation_contexts
from core.src.meta_model.optimize_parameters.fold_matrix_cache import build_fold_matrix_cache


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


class TestFoldMatrixCache:
    def test_uses_dmatrix_for_validation_and_quantile_for_train(
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

        assert cache is not None
        first_fold = cache[min(cache.keys())]
        assert isinstance(first_fold.validation_matrix, _FakeDMatrix)
        assert not isinstance(first_fold.validation_matrix, _FakeQuantileDMatrix)
        assert all(
            isinstance(window.train_matrix, _FakeQuantileDMatrix)
            for window in first_fold.train_windows
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
