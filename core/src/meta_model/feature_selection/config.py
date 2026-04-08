from __future__ import annotations

"""Configuration for the feature selection pipeline (folds, thresholds, proxy backtest)."""

from dataclasses import dataclass, field
from typing import Any

from core.src.meta_model.data.constants import RANDOM_SEED
from core.src.meta_model.runtime_parallelism import resolve_available_cpu_count

DEFAULT_SELECTION_FOLD_COUNT: int = 10
DEFAULT_GROUP_SAMPLE_SIZE: int = 50_000
DEFAULT_MAX_GROUP_SIZE: int = 64
DEFAULT_SEARCH_BEAM_WIDTH: int = 6
DEFAULT_PAIR_SEED_GROUP_LIMIT: int = 24
DEFAULT_POSITIVE_FOLD_SHARE_MIN: float = 0.50
DEFAULT_LOWER_QUARTILE_FOLD_PNL_FLOOR: float = -0.0005
DEFAULT_TURNOVER_GUARDRAIL_MULTIPLIER: float = 1.15
DEFAULT_MAX_DRAWDOWN_GUARDRAIL_ADDITIVE: float = 0.05
DEFAULT_NULL_BOOTSTRAP_COUNT: int = 32
DEFAULT_MAX_ACTIVE_MATRIX_GIB: float = 6.0
DEFAULT_PROXY_TRAINING_ROUNDS: int = 128
DEFAULT_EMIT_INPUT_INVENTORY: bool = True
DEFAULT_PROXY_TOP_FRACTION: float = 0.02
DEFAULT_PROXY_OPEN_HURDLE_BPS: float = 0.0
# Must match evaluate backtest (long-only selection); "none" is not a runtime mode there.
DEFAULT_PROXY_NEUTRALITY_MODE: str = "long_only"
DEFAULT_SFI_MIN_COVERAGE_FRACTION: float = 0.90
DEFAULT_LINEAR_CORRELATION_THRESHOLD: float = 0.80
DEFAULT_DISTANCE_CORRELATION_THRESHOLD: float = 0.80
DEFAULT_DISTANCE_CORRELATION_SAMPLE_SIZE: int = 512
DEFAULT_DISTANCE_CORRELATION_MAX_FEATURES: int = 384
DEFAULT_TARGET_DISTANCE_CORRELATION_THRESHOLD: float = 0.005
DEFAULT_TARGET_DISTANCE_CORRELATION_SAMPLE_SIZE: int = 2048
DEFAULT_TRAIN_SAMPLING_FRACTION: float = 1.0
DEFAULT_SELECTED_FEATURE_COUNT: int = 30


def _default_parallel_workers() -> int:
    return resolve_available_cpu_count()


def _build_default_proxy_xgboost_params() -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "seed": RANDOM_SEED,
        "verbosity": 0,
        "max_depth": 2,
        "eta": 0.03372328559359028,
        "min_child_weight": 4.611500076567313,
        "subsample": 0.6081400083136549,
        "colsample_bytree": 0.43551711140201593,
        "gamma": 0.002653309978526908,
        "lambda": 0.0172765054190332,
        "alpha": 0.12122829698297534,
        "max_bin": 246,
    }


@dataclass(frozen=True)
class FeatureSelectionConfig:
    """Immutable configuration for feature selection: folds, pruning thresholds, proxy model."""

    random_seed: int = RANDOM_SEED
    fold_count: int = DEFAULT_SELECTION_FOLD_COUNT
    group_sample_size: int = DEFAULT_GROUP_SAMPLE_SIZE
    max_group_size: int = DEFAULT_MAX_GROUP_SIZE
    parallel_workers: int = field(default_factory=_default_parallel_workers)
    state_evaluation_workers: int | None = None
    search_beam_width: int = DEFAULT_SEARCH_BEAM_WIDTH
    pair_seed_group_limit: int = DEFAULT_PAIR_SEED_GROUP_LIMIT
    positive_fold_share_min: float = DEFAULT_POSITIVE_FOLD_SHARE_MIN
    lower_quartile_fold_pnl_floor: float = DEFAULT_LOWER_QUARTILE_FOLD_PNL_FLOOR
    turnover_guardrail_multiplier: float = DEFAULT_TURNOVER_GUARDRAIL_MULTIPLIER
    max_drawdown_guardrail_additive: float = DEFAULT_MAX_DRAWDOWN_GUARDRAIL_ADDITIVE
    null_bootstrap_count: int = DEFAULT_NULL_BOOTSTRAP_COUNT
    max_active_matrix_gib: float = DEFAULT_MAX_ACTIVE_MATRIX_GIB
    proxy_training_rounds: int = DEFAULT_PROXY_TRAINING_ROUNDS
    proxy_top_fraction: float = DEFAULT_PROXY_TOP_FRACTION
    proxy_open_hurdle_bps: float = DEFAULT_PROXY_OPEN_HURDLE_BPS
    proxy_neutrality_mode: str = DEFAULT_PROXY_NEUTRALITY_MODE
    proxy_xgboost_params: dict[str, Any] = field(default_factory=_build_default_proxy_xgboost_params)
    sfi_min_coverage_fraction: float = DEFAULT_SFI_MIN_COVERAGE_FRACTION
    linear_correlation_threshold: float = DEFAULT_LINEAR_CORRELATION_THRESHOLD
    distance_correlation_threshold: float = DEFAULT_DISTANCE_CORRELATION_THRESHOLD
    distance_correlation_sample_size: int = DEFAULT_DISTANCE_CORRELATION_SAMPLE_SIZE
    distance_correlation_max_features: int = DEFAULT_DISTANCE_CORRELATION_MAX_FEATURES
    target_distance_correlation_threshold: float = DEFAULT_TARGET_DISTANCE_CORRELATION_THRESHOLD
    target_distance_correlation_sample_size: int = DEFAULT_TARGET_DISTANCE_CORRELATION_SAMPLE_SIZE
    train_sampling_fraction: float = DEFAULT_TRAIN_SAMPLING_FRACTION
    selected_feature_count: int = DEFAULT_SELECTED_FEATURE_COUNT
    emit_input_inventory: bool = DEFAULT_EMIT_INPUT_INVENTORY

    def resolved_state_evaluation_workers(self, *, fold_count: int) -> int:
        del fold_count
        if self.parallel_workers <= 0:
            raise ValueError("parallel_workers must be strictly positive.")
        if self.state_evaluation_workers is not None:
            return min(max(1, self.state_evaluation_workers), self.parallel_workers)
        return 1

    def resolved_fold_parallel_workers(self, *, fold_count: int) -> int:
        state_workers = self.resolved_state_evaluation_workers(fold_count=fold_count)
        per_state_budget = max(1, self.parallel_workers // max(1, state_workers))
        return min(max(1, fold_count), per_state_budget)

    def resolved_model_threads_per_worker(self, *, fold_count: int) -> int:
        state_workers = self.resolved_state_evaluation_workers(fold_count=fold_count)
        fold_workers = self.resolved_fold_parallel_workers(fold_count=fold_count)
        per_state_budget = max(1, self.parallel_workers // max(1, state_workers))
        return max(1, per_state_budget // max(1, fold_workers))
