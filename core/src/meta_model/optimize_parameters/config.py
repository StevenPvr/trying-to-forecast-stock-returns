from __future__ import annotations

from dataclasses import dataclass

from core.src.meta_model.data.constants import RANDOM_SEED
from core.src.meta_model.model_contract import (
    DATE_COLUMN as CONTRACT_DATE_COLUMN,
    LABEL_EMBARGO_DAYS,
    MODEL_TARGET_COLUMN,
    SPLIT_COLUMN as CONTRACT_SPLIT_COLUMN,
    TICKER_COLUMN as CONTRACT_TICKER_COLUMN,
    TRAIN_SPLIT_NAME as CONTRACT_TRAIN_SPLIT_NAME,
    VAL_SPLIT_NAME as CONTRACT_VAL_SPLIT_NAME,
)

DATE_COLUMN: str = CONTRACT_DATE_COLUMN
SPLIT_COLUMN: str = CONTRACT_SPLIT_COLUMN
TICKER_COLUMN: str = CONTRACT_TICKER_COLUMN
TRAIN_SPLIT_NAME: str = CONTRACT_TRAIN_SPLIT_NAME
VAL_SPLIT_NAME: str = CONTRACT_VAL_SPLIT_NAME
TARGET_COLUMN: str = MODEL_TARGET_COLUMN
DEFAULT_FOLD_COUNT: int = 5
DEFAULT_TRIAL_COUNT: int = 200
TARGET_HORIZON_DAYS: int = LABEL_EMBARGO_DAYS
STABILITY_PENALTY_ALPHA: float = 0.10
TRAIN_WINDOW_STABILITY_ALPHA: float = 0.05
COMPLEXITY_PENALTY_ALPHA: float = 0.01
OBJECTIVE_STANDARD_ERROR_BOOTSTRAP_SAMPLES: int = 1024
OPTUNA_STUDY_NAME: str = "xgboost_walk_forward_daily_rank_ic"
EARLY_STOPPING_ROUNDS: int = 100
DEFAULT_BOOST_ROUNDS: int = 3000
RECENT_TRAIN_TAIL_FRACTION: float = 0.67
RANDOM_TRAIN_WINDOW_COUNT: int = 1
RANDOM_TRAIN_WINDOW_MIN_FRACTION: float = 0.60


@dataclass(frozen=True)
class OptimizationConfig:
    random_seed: int = RANDOM_SEED
    fold_count: int = DEFAULT_FOLD_COUNT
    trial_count: int = DEFAULT_TRIAL_COUNT
    target_horizon_days: int = TARGET_HORIZON_DAYS
    stability_penalty_alpha: float = STABILITY_PENALTY_ALPHA
    train_window_stability_alpha: float = TRAIN_WINDOW_STABILITY_ALPHA
    complexity_penalty_alpha: float = COMPLEXITY_PENALTY_ALPHA
    objective_standard_error_bootstrap_samples: int = OBJECTIVE_STANDARD_ERROR_BOOTSTRAP_SAMPLES
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS
    boost_rounds: int = DEFAULT_BOOST_ROUNDS
    recent_train_tail_fraction: float = RECENT_TRAIN_TAIL_FRACTION
    random_train_window_count: int = RANDOM_TRAIN_WINDOW_COUNT
    random_train_window_min_fraction: float = RANDOM_TRAIN_WINDOW_MIN_FRACTION
