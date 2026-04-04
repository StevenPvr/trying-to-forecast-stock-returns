from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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


def _make_preprocessed_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, start, periods in (
        ("train", "2018-01-01", 20),
        ("val", "2019-02-01", 10),
    ):
        for idx, date in enumerate(pd.date_range(start, periods=periods, freq="B"), start=1):
            for ticker, offset in (("AAA", 0.0), ("BBB", 1.0)):
                rows.append({
                    "date": date,
                    "ticker": ticker,
                    "target_main": 0.01 * idx + offset * 0.001,
                    MODEL_TARGET_COLUMN: 0.01 * idx + offset * 0.001,
                    "dataset_split": split_name,
                    "feature_a": float(idx + offset),
                    "feature_b": float((idx + offset) * 2),
                })
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


class TestBuildFoldEvaluationContexts:
    def test_precomputes_train_windows_and_validation_metric_context_per_fold(self) -> None:
        data = _make_preprocessed_df()
        bundle = build_optimization_dataset_bundle(data, dataset_path=Path("synthetic.parquet"))
        config = OptimizationConfig(fold_count=5, target_horizon_days=1)
        folds = build_walk_forward_folds(
            bundle.metadata,
            fold_count=config.fold_count,
            target_horizon_days=config.target_horizon_days,
        )

        contexts = build_fold_evaluation_contexts(bundle, folds, config)

        assert len(contexts) == len(folds)
        assert contexts[0].fold.index == 1
        assert contexts[0].train_windows[0].label == "full"
        assert len(contexts[0].validation_rank_ic_context.groups) >= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
