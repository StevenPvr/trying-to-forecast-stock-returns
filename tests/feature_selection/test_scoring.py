from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.feature_selection.config import FeatureSelectionConfig
from core.src.meta_model.feature_selection.cv import SelectionFold
from core.src.meta_model.feature_selection.objective import FoldEconomicScore
from core.src.meta_model.feature_selection.scoring import BacktestFeatureSubsetScorer
from core.src.meta_model.model_contract import DATE_COLUMN, MODEL_TARGET_COLUMN, SPLIT_COLUMN, TICKER_COLUMN


class _DummyCache:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def build_feature_frame(self, feature_names: list[str]) -> pd.DataFrame:
        selected_columns = [DATE_COLUMN, TICKER_COLUMN, SPLIT_COLUMN, MODEL_TARGET_COLUMN, *feature_names]
        feature_frame = cast(pd.DataFrame, self._frame.loc[:, selected_columns].copy())
        return pd.DataFrame(feature_frame)


def _make_fold(index: int, train_indices: list[int], validation_indices: list[int]) -> SelectionFold:
    return SelectionFold(
        index=index,
        weight=float(index),
        train_indices=np.asarray(train_indices, dtype=np.int64),
        validation_indices=np.asarray(validation_indices, dtype=np.int64),
        train_end_date=cast(pd.Timestamp, pd.Timestamp("2020-01-10")),
        validation_start_date=cast(pd.Timestamp, pd.Timestamp("2020-01-13")),
        validation_end_date=cast(pd.Timestamp, pd.Timestamp("2020-01-14")),
    )


def test_backtest_feature_subset_scorer_parallelizes_fold_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({
        DATE_COLUMN: pd.date_range("2020-01-01", periods=16, freq="B"),
        TICKER_COLUMN: ["AAA"] * 16,
        SPLIT_COLUMN: ["train"] * 16,
        MODEL_TARGET_COLUMN: np.linspace(-1.0, 1.0, 16),
        "feature_signal": np.linspace(0.0, 1.0, 16),
    })
    folds = [
        _make_fold(1, list(range(0, 8)), list(range(8, 10))),
        _make_fold(2, list(range(0, 9)), list(range(9, 11))),
        _make_fold(3, list(range(0, 10)), list(range(10, 12))),
        _make_fold(4, list(range(0, 11)), list(range(11, 13))),
    ]

    def slow_fold_score(
        train_frame: pd.DataFrame,
        feature_names: list[str],
        fold: SelectionFold,
        model_spec: object,
        backtest_config: object,
        cost_config: object,
    ) -> FoldEconomicScore:
        del train_frame, feature_names, model_spec, backtest_config, cost_config
        time.sleep(0.2)
        return FoldEconomicScore(
            index=fold.index,
            weight=fold.weight,
            net_pnl_after_costs=0.01 * fold.index,
            alpha_over_benchmark_net=0.01 * fold.index,
            turnover_annualized=0.40,
            max_drawdown=-0.05,
            daily_rank_ic_mean=0.01 * fold.index,
            daily_rank_ic_ir=0.01 * fold.index,
            daily_top_bottom_spread_mean=0.01 * fold.index,
        )

    monkeypatch.setattr(
        "core.src.meta_model.feature_selection.scoring._score_validation_fold",
        slow_fold_score,
    )

    scorer = BacktestFeatureSubsetScorer(
        cast(Any, _DummyCache(frame)),
        folds,
        FeatureSelectionConfig(parallel_workers=4, state_evaluation_workers=1, null_bootstrap_count=0),
    )

    start_time = time.perf_counter()
    score = scorer(["feature_signal"])
    elapsed = time.perf_counter() - start_time

    assert elapsed < 0.55
    assert score.objective_score > 0.0


def test_backtest_feature_subset_scorer_reuses_same_cost_config_across_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({
        DATE_COLUMN: pd.date_range("2020-01-01", periods=8, freq="B"),
        TICKER_COLUMN: ["AAA"] * 8,
        SPLIT_COLUMN: ["train"] * 8,
        MODEL_TARGET_COLUMN: np.linspace(-1.0, 1.0, 8),
        "feature_signal": np.linspace(0.0, 1.0, 8),
    })
    folds = [
        _make_fold(1, list(range(0, 4)), list(range(4, 6))),
        _make_fold(2, list(range(0, 5)), list(range(5, 7))),
    ]
    observed_cost_config_ids: list[int] = []

    def score_with_cost_config_capture(
        train_frame: pd.DataFrame,
        feature_names: list[str],
        fold: SelectionFold,
        model_spec: object,
        backtest_config: object,
        cost_config: object,
    ) -> FoldEconomicScore:
        del train_frame, feature_names, model_spec, backtest_config
        observed_cost_config_ids.append(id(cost_config))
        return FoldEconomicScore(
            index=fold.index,
            weight=fold.weight,
            net_pnl_after_costs=0.01,
            alpha_over_benchmark_net=0.01,
            turnover_annualized=0.40,
            max_drawdown=-0.05,
            daily_rank_ic_mean=0.01,
            daily_rank_ic_ir=0.01,
            daily_top_bottom_spread_mean=0.01,
        )

    monkeypatch.setattr(
        "core.src.meta_model.feature_selection.scoring._score_validation_fold",
        score_with_cost_config_capture,
    )

    scorer = BacktestFeatureSubsetScorer(
        cast(Any, _DummyCache(frame)),
        folds,
        FeatureSelectionConfig(parallel_workers=1, null_bootstrap_count=0),
    )

    score = scorer(["feature_signal"])

    assert score.objective_score > 0.0
    assert len(observed_cost_config_ids) == 2
    assert len(set(observed_cost_config_ids)) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
