from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.src.meta_model.optimize_parameters.config import OptimizationConfig
from core.src.meta_model.optimize_parameters.cv import WalkForwardFold
from core.src.meta_model.optimize_parameters.dataset import OptimizationDatasetBundle
from core.src.meta_model.optimize_parameters.metric_context import (
    DailyRankIcContext,
    build_daily_rank_ic_context,
)
from core.src.meta_model.optimize_parameters.robustness import TrainWindow, build_train_windows


@dataclass(frozen=True)
class FoldEvaluationContext:
    fold: WalkForwardFold
    train_windows: tuple[TrainWindow, ...]
    validation_rank_ic_context: DailyRankIcContext


def build_fold_evaluation_contexts(
    dataset_bundle: OptimizationDatasetBundle,
    folds: list[WalkForwardFold],
    optimization_config: OptimizationConfig,
) -> list[FoldEvaluationContext]:
    contexts: list[FoldEvaluationContext] = []
    for fold in folds:
        validation_dates = dataset_bundle.metadata.iloc[fold.validation_indices]["date"].to_numpy()
        contexts.append(
            FoldEvaluationContext(
                fold=fold,
                train_windows=tuple(
                    build_train_windows(
                        dataset_bundle.metadata,
                        fold.train_indices,
                        random_seed=optimization_config.random_seed + fold.index,
                        recent_tail_fraction=optimization_config.recent_train_tail_fraction,
                        random_window_count=optimization_config.random_train_window_count,
                        random_window_min_fraction=optimization_config.random_train_window_min_fraction,
                    ),
                ),
                validation_rank_ic_context=build_daily_rank_ic_context(
                    np.asarray(dataset_bundle.target_array[fold.validation_indices], dtype=np.float64),
                    np.asarray(validation_dates),
                ),
            ),
        )
    return contexts
