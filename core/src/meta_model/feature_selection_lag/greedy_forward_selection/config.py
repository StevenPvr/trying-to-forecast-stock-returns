from __future__ import annotations

from dataclasses import dataclass

from core.src.meta_model.data.constants import RANDOM_SEED

TARGET_COLUMN: str = "target_main"
TRAIN_SPLIT_NAME: str = "train"
TRAIN_SAMPLE_FRACTION: float = 0.5
TRAIN_HOLDOUT_FRACTION: float = 0.2
CANDIDATE_FEATURE_NAME_COLUMN: str = "candidate_feature_name"


@dataclass(frozen=True)
class SFIModelConfig:
    random_seed: int = RANDOM_SEED
    allow_writing_files: bool = False
    iterations: int = 3000
    depth: int = 4
    learning_rate: float = 0.03
    loss_function: str = "RMSE"
    early_stopping_rounds: int = 100


__all__ = [
    "CANDIDATE_FEATURE_NAME_COLUMN",
    "SFIModelConfig",
    "TARGET_COLUMN",
    "TRAIN_HOLDOUT_FRACTION",
    "TRAIN_SAMPLE_FRACTION",
    "TRAIN_SPLIT_NAME",
]
