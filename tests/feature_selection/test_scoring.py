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

from core.src.meta_model.evaluate.config import BacktestConfig
from core.src.meta_model.evaluate.backtest import XtbCostConfig
from core.src.meta_model.broker_xtb.specs import BrokerSpecProvider, XtbInstrumentSpec
from core.src.meta_model.feature_selection.config import FeatureSelectionConfig
from core.src.meta_model.feature_selection.cv import SelectionFold
from core.src.meta_model.feature_selection.objective import FoldEconomicScore
from core.src.meta_model.feature_selection.scoring import BacktestFeatureSubsetScorer
from core.src.meta_model.model_contract import (
    DATE_COLUMN,
    MODEL_TARGET_COLUMN,
    REALIZED_RETURN_COLUMN,
    SPLIT_COLUMN,
    TICKER_COLUMN,
)


class _DummyCache:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def build_feature_frame(
        self,
        feature_names: list[str],
        *,
        row_indices: np.ndarray | None = None,
    ) -> pd.DataFrame:
        context_columns = [
            DATE_COLUMN,
            TICKER_COLUMN,
            SPLIT_COLUMN,
            MODEL_TARGET_COLUMN,
            REALIZED_RETURN_COLUMN,
            "company_sector",
            "stock_open_price",
            "stock_trading_volume",
        ]
        selected_columns = [
            column_name
            for column_name in [*context_columns, *feature_names]
            if column_name in self._frame.columns
        ]
        feature_frame = cast(pd.DataFrame, self._frame.loc[:, selected_columns].copy())
        if row_indices is None:
            return pd.DataFrame(feature_frame)
        return pd.DataFrame(feature_frame.take(np.asarray(row_indices, dtype=np.int64)).reset_index(drop=True))

    def get_feature_array(self, feature_name: str) -> np.ndarray:
        feature_series = cast(pd.Series, self._frame[feature_name])
        return feature_series.to_numpy(dtype=np.float32, copy=True)


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


def _build_test_cost_config() -> XtbCostConfig:
    provider = BrokerSpecProvider(
        specs=(
            XtbInstrumentSpec(
                symbol="AAA",
                instrument_group="stock_cash",
                currency="EUR",
                spread_bps=0.0,
                slippage_bps=0.0,
                long_swap_bps_daily=0.0,
                short_swap_bps_daily=0.0,
                margin_requirement=1.0,
                max_adv_participation=0.05,
                effective_from="2000-01-01",
                fx_conversion_bps=0.0,
            ),
            XtbInstrumentSpec(
                symbol="BBB",
                instrument_group="stock_cash",
                currency="EUR",
                spread_bps=0.0,
                slippage_bps=0.0,
                long_swap_bps_daily=0.0,
                short_swap_bps_daily=0.0,
                margin_requirement=1.0,
                max_adv_participation=0.05,
                effective_from="2000-01-01",
                fx_conversion_bps=0.0,
            ),
        ),
        fallback_to_defaults=False,
    )
    return XtbCostConfig(account_currency="EUR", broker_spec_provider=provider)


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
        fold_train: pd.DataFrame,
        fold_validation: pd.DataFrame,
        feature_names: list[str],
        fold: SelectionFold,
        model_spec: object,
        backtest_config: object,
        cost_config: object,
        optimizer_artifacts: object,
    ) -> FoldEconomicScore:
        del fold_train, fold_validation, feature_names, model_spec, backtest_config, cost_config, optimizer_artifacts
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
        FeatureSelectionConfig(parallel_workers=4, null_bootstrap_count=0),
    )

    start_time = time.perf_counter()
    score = scorer(["feature_signal"])
    elapsed = time.perf_counter() - start_time

    assert elapsed < 0.55
    assert score.objective_score > 0.0


def test_feature_selection_config_default_parallel_budget_uses_all_cores() -> None:
    config = FeatureSelectionConfig(parallel_workers=12, null_bootstrap_count=0)

    assert config.resolved_state_evaluation_workers(fold_count=4) == 1
    assert config.resolved_fold_parallel_workers(fold_count=4) == 4
    assert config.resolved_model_threads_per_worker(fold_count=4) == 3


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
        fold_train: pd.DataFrame,
        fold_validation: pd.DataFrame,
        feature_names: list[str],
        fold: SelectionFold,
        model_spec: object,
        backtest_config: object,
        cost_config: object,
        optimizer_artifacts: object,
    ) -> FoldEconomicScore:
        del fold_train, fold_validation, feature_names, model_spec, backtest_config, optimizer_artifacts
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


def test_backtest_feature_subset_scorer_builds_proxy_optimizer_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    frame = pd.DataFrame(
        {
            DATE_COLUMN: np.repeat(dates, 2),
            TICKER_COLUMN: ["AAA", "BBB"] * len(dates),
            SPLIT_COLUMN: ["train"] * (2 * len(dates)),
            MODEL_TARGET_COLUMN: [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.85, 0.15, 0.75, 0.25, 0.65, 0.35],
            REALIZED_RETURN_COLUMN: [0.04, 0.01, 0.03, 0.00, 0.02, -0.01, 0.05, 0.01, 0.04, 0.00, 0.03, -0.01],
            "stock_open_price": [100.0] * (2 * len(dates)),
            "stock_trading_volume": [1_000_000.0] * (2 * len(dates)),
            "company_sector": ["Tech", "Finance"] * len(dates),
            "feature_signal": np.linspace(0.0, 1.0, 2 * len(dates)),
        },
    )
    folds = [_make_fold(1, list(range(0, 8)), list(range(8, 12)))]

    monkeypatch.setattr(
        "core.src.meta_model.feature_selection.scoring.fit_model",
        lambda model_spec, train_frame, feature_names: object(),
    )
    monkeypatch.setattr(
        "core.src.meta_model.feature_selection.scoring.predict_model",
        lambda artifact, validation_frame, feature_names: np.where(
            validation_frame[TICKER_COLUMN].to_numpy() == "AAA",
            0.03,
            0.01,
        ),
    )

    scorer = BacktestFeatureSubsetScorer(
        cast(Any, _DummyCache(frame)),
        folds,
        FeatureSelectionConfig(parallel_workers=1, null_bootstrap_count=0),
        backtest_config=BacktestConfig(
            top_fraction=0.5,
            open_hurdle_bps=0.0,
            hold_period_days=1,
            lambda_risk=0.0,
            lambda_turnover=0.0,
            lambda_cost=0.0,
            max_position_weight=0.5,
            max_sector_weight=1.0,
            miqp_candidate_pool_size=2,
        ),
    )
    scorer._cost_config = _build_test_cost_config()

    score = scorer(["feature_signal"])

    assert score.objective_score > 0.0
    assert len(score.fold_scores) == 1
    assert np.isfinite(score.fold_scores[0].net_pnl_after_costs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
