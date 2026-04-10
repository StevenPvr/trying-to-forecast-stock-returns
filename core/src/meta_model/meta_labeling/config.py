from __future__ import annotations

"""Configuration for the sequential meta-labeling stage."""

from dataclasses import dataclass
import os

from core.src.meta_model.data.constants import RANDOM_SEED
from core.src.meta_model.model_contract import TRAIN_SPLIT_NAME


@dataclass(frozen=True)
class MetaLabelingConfig:
    random_seed: int = RANDOM_SEED
    fit_split_name: str = TRAIN_SPLIT_NAME
    burn_in_train_fraction: float = 0.20
    primary_oos_parallel_workers: int = max(1, os.cpu_count() or 1)
    fold_count: int = 4
    boost_rounds: int = 2000
    early_stopping_rounds: int = 100
    early_stopping_validation_fraction: float = 0.15
    minimum_training_rounds: int = 25
    meta_primary_candidate_threshold: float = 0.0
    meta_min_target_net_return: float = 0.010
    meta_decision_threshold: float = 0.5
    meta_mcc_use_balanced_class_weights: bool = True
    meta_trial_count: int = 100
    meta_optuna_startup_trials: int = 20
    meta_target_column: str = "meta_label"
    meta_probability_column: str = "meta_probability"
    calibration_holdout_fraction: float = 0.20
    refinement_strategy: str = "binary_gate"
    soft_shifted_floor: float = 0.45
    rank_blend_lambda: float = 0.50
