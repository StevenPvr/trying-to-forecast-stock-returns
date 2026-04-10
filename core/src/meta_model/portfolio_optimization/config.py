from __future__ import annotations

"""Configuration for train-only portfolio optimisation."""

from dataclasses import dataclass

from core.src.meta_model.data.constants import RANDOM_SEED
from core.src.meta_model.model_contract import TRAIN_SPLIT_NAME


@dataclass(frozen=True)
class PortfolioOptimizationConfig:
    random_seed: int = RANDOM_SEED
    fit_split_name: str = TRAIN_SPLIT_NAME
    enforce_train_only_fit: bool = True
    oof_fold_count: int = 5
    oof_parallel_workers: int | None = None
    oof_min_train_dates: int = 60
    trial_parallel_workers: int | None = None
    trial_progress_log_every_days: int = 250
    covariance_lookback_days: int = 63
    covariance_min_history_days: int = 40
    max_position_weight: float = 0.02
    max_sector_weight: float = 0.20
    no_trade_buffer_bps: float = 15.0
    lambda_risk: float = 3.0
    lambda_turnover: float = 0.001
    lambda_cost: float = 1.0
    min_target_weight: float = 0.001
    miqp_time_limit_seconds: float = 2.0
    miqp_relative_gap: float = 0.005
    miqp_candidate_pool_size: int = 40
    miqp_primary_objective_tolerance_bps: float = 0.5
    lambda_risk_grid: tuple[float, ...] = (1.0, 5.0)
    lambda_turnover_grid: tuple[float, ...] = (5e-4, 1e-3)
    no_trade_buffer_bps_grid: tuple[float, ...] = (5.0, 15.0)
    max_position_weight_grid: tuple[float, ...] = (0.01, 0.02)
    max_sector_weight_grid: tuple[float, ...] = (0.15, 0.20)
    covariance_lookback_days_grid: tuple[int, ...] = (63, 126)
