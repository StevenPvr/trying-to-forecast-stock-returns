from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
import logging
from threading import RLock
from typing import cast

import numpy as np
import pandas as pd

from core.src.meta_model.evaluate.backtest import (
    BacktestState,
    XtbCostConfig,
    finalize_backtest_state,
    process_prediction_day,
)
from core.src.meta_model.evaluate.config import BacktestConfig
from core.src.meta_model.feature_selection.cache import FeatureSelectionRuntimeCache
from core.src.meta_model.feature_selection.config import FeatureSelectionConfig
from core.src.meta_model.feature_selection.cv import SelectionFold
from core.src.meta_model.feature_selection.objective import (
    FoldEconomicScore,
    SubsetEconomicScore,
    aggregate_subset_score,
)
from core.src.meta_model.research_metrics import (
    build_daily_signal_diagnostics,
    summarize_daily_signal_diagnostics,
)
from core.src.meta_model.model_contract import DATE_COLUMN, MODEL_TARGET_COLUMN, PREDICTION_COLUMN
from core.src.meta_model.model_registry.main import ModelSpec, fit_model, predict_model

LOGGER: logging.Logger = logging.getLogger(__name__)


class BacktestFeatureSubsetScorer:
    def __init__(
        self,
        cache: FeatureSelectionRuntimeCache,
        folds: list[SelectionFold],
        config: FeatureSelectionConfig,
        *,
        backtest_config: BacktestConfig | None = None,
    ) -> None:
        if config.parallel_workers <= 0:
            raise ValueError("parallel_workers must be strictly positive.")
        self._cache = cache
        self._folds = folds
        self._config = config
        self._backtest_config = backtest_config or BacktestConfig(
            top_fraction=config.proxy_top_fraction,
            neutrality_mode=config.proxy_neutrality_mode,
            open_hurdle_bps=config.proxy_open_hurdle_bps,
        )
        self._state_worker_budget = config.resolved_state_evaluation_workers(fold_count=len(folds))
        self._fold_worker_count = config.resolved_fold_parallel_workers(fold_count=len(folds))
        self._model_thread_count = config.resolved_model_threads_per_worker(fold_count=len(folds))
        self._cost_config = XtbCostConfig()
        model_params = dict(config.proxy_xgboost_params)
        model_params.setdefault("nthread", self._model_thread_count)
        self._model_spec = ModelSpec(
            model_name="xgboost",
            params=model_params,
            training_rounds=config.proxy_training_rounds,
        )
        self._memo: dict[tuple[str, ...], SubsetEconomicScore] = {}
        self._memo_lock = RLock()
        LOGGER.info(
            "Feature subset scorer configured: folds=%d | state_workers_budget=%d | fold_workers=%d | model_threads_per_worker=%d | proxy_top_fraction=%.4f | proxy_open_hurdle_bps=%.2f | proxy_neutrality_mode=%s | proxy_apply_prediction_hurdle=%s",
            len(folds),
            self._state_worker_budget,
            self._fold_worker_count,
            self._model_thread_count,
            self._backtest_config.top_fraction,
            self._backtest_config.open_hurdle_bps,
            self._backtest_config.neutrality_mode,
            self._backtest_config.apply_prediction_hurdle,
        )

    def __call__(self, feature_names: list[str]) -> SubsetEconomicScore:
        score_key = tuple(sorted(set(feature_names)))
        with self._memo_lock:
            cached_score = self._memo.get(score_key)
        if cached_score is not None:
            return cached_score
        subset_score = self._score_feature_subset(list(score_key))
        with self._memo_lock:
            self._memo[score_key] = subset_score
        return subset_score

    def _score_feature_subset(self, feature_names: list[str]) -> SubsetEconomicScore:
        if not feature_names:
            return aggregate_subset_score([], [])
        train_frame = self._cache.build_feature_frame(feature_names)
        if self._fold_worker_count == 1 or len(self._folds) == 1:
            fold_scores = [self._score_single_fold(train_frame, feature_names, fold) for fold in self._folds]
        else:
            with ThreadPoolExecutor(max_workers=self._fold_worker_count, thread_name_prefix="feature-selection") as executor:
                fold_scores = list(
                    executor.map(
                        self._score_single_fold_from_shared_frame,
                        repeat(train_frame, len(self._folds)),
                        repeat(feature_names, len(self._folds)),
                        self._folds,
                    ),
                )
        return aggregate_subset_score(feature_names, fold_scores)

    def _score_single_fold_from_shared_frame(
        self,
        train_frame: pd.DataFrame,
        feature_names: list[str],
        fold: SelectionFold,
    ) -> FoldEconomicScore:
        return self._score_single_fold(train_frame, feature_names, fold)

    def _score_single_fold(
        self,
        train_frame: pd.DataFrame,
        feature_names: list[str],
        fold: SelectionFold,
    ) -> FoldEconomicScore:
        return _score_validation_fold(
            train_frame,
            feature_names,
            fold,
            self._model_spec,
            self._backtest_config,
            self._cost_config,
        )


def _score_validation_fold(
    train_frame: pd.DataFrame,
    feature_names: list[str],
    fold: SelectionFold,
    model_spec: ModelSpec,
    backtest_config: BacktestConfig,
    cost_config: XtbCostConfig,
) -> FoldEconomicScore:
    fold_train = pd.DataFrame(train_frame.take(fold.train_indices).reset_index(drop=True))
    fold_validation = pd.DataFrame(train_frame.take(fold.validation_indices).reset_index(drop=True))
    artifact = fit_model(model_spec, fold_train, feature_names)
    predictions = predict_model(artifact, fold_validation, feature_names)
    predicted_validation = fold_validation.copy()
    predicted_validation[PREDICTION_COLUMN] = predictions
    summary = _score_predicted_validation(
        predicted_validation,
        backtest_config,
        cost_config,
        target_column=model_spec.target_column,
    )
    return FoldEconomicScore(
        index=fold.index,
        weight=fold.weight,
        net_pnl_after_costs=float(summary["net_pnl_after_costs"]),
        alpha_over_benchmark_net=float(summary["alpha_over_benchmark_net"]),
        turnover_annualized=float(summary["turnover_annualized"]),
        max_drawdown=float(summary["max_drawdown"]),
        daily_rank_ic_mean=float(summary["daily_rank_ic_mean"]),
        daily_rank_ic_ir=float(summary["daily_rank_ic_ir"]),
        daily_top_bottom_spread_mean=float(summary["daily_top_bottom_spread_mean"]),
    )


def _score_predicted_validation(
    predicted_validation: pd.DataFrame,
    backtest_config: BacktestConfig,
    cost_config: XtbCostConfig,
    *,
    target_column: str = MODEL_TARGET_COLUMN,
) -> dict[str, float]:
    state = BacktestState()
    date_series = cast(pd.Series, predicted_validation[DATE_COLUMN])
    unique_dates = pd.Index(pd.to_datetime(date_series).drop_duplicates().sort_values())
    for daily_predictions in _iter_prediction_days(predicted_validation):
        process_prediction_day(
            state=state,
            daily_predictions=daily_predictions,
            unique_dates=unique_dates,
            top_fraction=backtest_config.top_fraction,
            allocation_fraction=backtest_config.allocation_fraction,
            action_cap_fraction=backtest_config.action_cap_fraction,
            gross_cap_fraction=backtest_config.gross_cap_fraction,
            adv_participation_limit=backtest_config.adv_participation_limit,
            neutrality_mode=backtest_config.neutrality_mode,
            open_hurdle_bps=backtest_config.open_hurdle_bps,
            apply_prediction_hurdle=backtest_config.apply_prediction_hurdle,
            hold_period_days=backtest_config.hold_period_days,
            cost_config=cost_config,
            logger=None,
        )
    _, _, summary = finalize_backtest_state(state)
    daily_signal_diagnostics = build_daily_signal_diagnostics(
        predicted_validation,
        target_column=target_column,
        top_fraction=backtest_config.top_fraction,
    )
    signal_summary = summarize_daily_signal_diagnostics(daily_signal_diagnostics)
    return {**summary, **signal_summary}


def score_predicted_validation(
    predicted_validation: pd.DataFrame,
    backtest_config: BacktestConfig,
    cost_config: XtbCostConfig,
    *,
    target_column: str = MODEL_TARGET_COLUMN,
) -> dict[str, float]:
    return _score_predicted_validation(
        predicted_validation,
        backtest_config,
        cost_config,
        target_column=target_column,
    )


def _iter_prediction_days(predicted_validation: pd.DataFrame) -> Iterator[pd.DataFrame]:
    if predicted_validation.empty:
        return
    date_values = pd.to_datetime(cast(pd.Series, predicted_validation[DATE_COLUMN])).to_numpy(copy=False)
    if date_values.size == 0:
        return
    boundary_positions = np.flatnonzero(date_values[1:] != date_values[:-1]) + 1
    start_position = 0
    for stop_position in np.concatenate((boundary_positions, np.asarray([len(predicted_validation)], dtype=np.int64))):
        daily_frame = cast(pd.DataFrame, predicted_validation.iloc[start_position:int(stop_position)].reset_index(drop=True))
        yield pd.DataFrame(daily_frame)
        start_position = int(stop_position)


__all__ = ["BacktestFeatureSubsetScorer", "score_predicted_validation"]
