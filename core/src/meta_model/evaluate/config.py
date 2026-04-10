from __future__ import annotations

"""Backtest configuration defaults and validation for the evaluate pipeline."""

from dataclasses import dataclass

from core.src.meta_model.model_contract import (
    EXECUTION_LAG_DAYS,
    HOLD_PERIOD_DAYS,
    MODEL_TARGET_COLUMN,
)

TOP_FRACTION: float = 0.01
ACTION_CAP_FRACTION: float = 0.05
ALLOCATION_FRACTION: float = 0.05
GROSS_CAP_FRACTION: float = 1.0
TARGET_COLUMN: str = MODEL_TARGET_COLUMN
BENCHMARK_MODE: str = "universe_equal_weight"
NEUTRALITY_MODE: str = "long_only"
ACCOUNT_CURRENCY: str = "EUR"
STARTING_CASH_EUR: float = 10_000.0
ADV_PARTICIPATION_LIMIT: float = 0.05
MAX_SPREAD_BPS: float = 35.0
OPEN_HURDLE_BPS: float = 12.0
HOLD_HURDLE_BPS: float = 6.0
PBO_MAX_THRESHOLD: float = 0.20
DSR_MIN_THRESHOLD: float = 0.10
APPLY_PREDICTION_HURDLE: bool = False
EVALUATION_TRAINING_MODE: str = "walk_forward"
PORTFOLIO_CONSTRUCTION_MODE: str = "optimizer_miqp"
SOLVER_BACKEND: str = "scip_miqp"
ALPHA_CALIBRATION_METHOD: str = "identity"
COVARIANCE_LOOKBACK_DAYS: int = 63
COVARIANCE_MIN_HISTORY_DAYS: int = 40
LAMBDA_RISK: float = 3.0
LAMBDA_TURNOVER: float = 0.001
LAMBDA_COST: float = 1.0
MAX_POSITION_WEIGHT: float = 0.02
MAX_SECTOR_WEIGHT: float = 0.20
NO_TRADE_BUFFER_BPS: float = 15.0
SOLVER_WARMUP_DAYS: int = 60
MIN_TARGET_WEIGHT: float = 0.001
MIQP_TIME_LIMIT_SECONDS: float = 2.0
MIQP_RELATIVE_GAP: float = 0.005
MIQP_CANDIDATE_POOL_SIZE: int = 40
MIQP_PRIMARY_OBJECTIVE_TOLERANCE_BPS: float = 0.5


@dataclass(frozen=True)
class BacktestConfig:
    """Immutable configuration for the long-only XTB cash-equity backtest."""

    top_fraction: float = TOP_FRACTION
    action_cap_fraction: float = ACTION_CAP_FRACTION
    allocation_fraction: float = ALLOCATION_FRACTION
    gross_cap_fraction: float = GROSS_CAP_FRACTION
    hold_period_days: int = HOLD_PERIOD_DAYS
    execution_lag_days: int = EXECUTION_LAG_DAYS
    benchmark_mode: str = BENCHMARK_MODE
    neutrality_mode: str = NEUTRALITY_MODE
    account_currency: str = ACCOUNT_CURRENCY
    starting_cash_eur: float = STARTING_CASH_EUR
    adv_participation_limit: float = ADV_PARTICIPATION_LIMIT
    max_spread_bps: float = MAX_SPREAD_BPS
    open_hurdle_bps: float = OPEN_HURDLE_BPS
    hold_hurdle_bps: float = HOLD_HURDLE_BPS
    apply_prediction_hurdle: bool = APPLY_PREDICTION_HURDLE
    pbo_max_threshold: float = PBO_MAX_THRESHOLD
    dsr_min_threshold: float = DSR_MIN_THRESHOLD
    portfolio_construction_mode: str = PORTFOLIO_CONSTRUCTION_MODE
    evaluation_training_mode: str = EVALUATION_TRAINING_MODE
    solver_backend: str = SOLVER_BACKEND
    alpha_calibration_method: str = ALPHA_CALIBRATION_METHOD
    covariance_lookback_days: int = COVARIANCE_LOOKBACK_DAYS
    covariance_min_history_days: int = COVARIANCE_MIN_HISTORY_DAYS
    lambda_risk: float = LAMBDA_RISK
    lambda_turnover: float = LAMBDA_TURNOVER
    lambda_cost: float = LAMBDA_COST
    max_position_weight: float = MAX_POSITION_WEIGHT
    max_sector_weight: float = MAX_SECTOR_WEIGHT
    no_trade_buffer_bps: float = NO_TRADE_BUFFER_BPS
    solver_warmup_days: int = SOLVER_WARMUP_DAYS
    min_target_weight: float = MIN_TARGET_WEIGHT
    miqp_time_limit_seconds: float = MIQP_TIME_LIMIT_SECONDS
    miqp_relative_gap: float = MIQP_RELATIVE_GAP
    miqp_candidate_pool_size: int = MIQP_CANDIDATE_POOL_SIZE
    miqp_primary_objective_tolerance_bps: float = MIQP_PRIMARY_OBJECTIVE_TOLERANCE_BPS


def validate_backtest_config(config: BacktestConfig) -> None:
    """Raise ``ValueError`` if any config constraint is violated."""
    if config.execution_lag_days < 0:
        raise ValueError("execution_lag_days must be non-negative.")
    if not config.benchmark_mode.strip():
        raise ValueError("benchmark_mode must be configured explicitly.")
    if not config.neutrality_mode.strip():
        raise ValueError("neutrality_mode must be configured explicitly.")
    if config.neutrality_mode != "long_only":
        raise ValueError("neutrality_mode must remain long_only for XTB cash equities.")
    if not config.account_currency.strip():
        raise ValueError("account_currency must be configured explicitly.")
    if config.starting_cash_eur <= 0.0:
        raise ValueError("starting_cash_eur must be strictly positive.")
    if config.max_spread_bps <= 0.0:
        raise ValueError("max_spread_bps must be strictly positive.")
    if config.hold_hurdle_bps >= config.open_hurdle_bps:
        raise ValueError("hold_hurdle_bps must remain strictly below open_hurdle_bps.")
    if config.portfolio_construction_mode != "optimizer_miqp":
        raise ValueError("portfolio_construction_mode must be optimizer_miqp.")
    if config.evaluation_training_mode not in {"walk_forward", "frozen_train_only"}:
        raise ValueError("evaluation_training_mode must be one of: walk_forward, frozen_train_only.")
    if config.solver_backend != "scip_miqp":
        raise ValueError("solver_backend must be scip_miqp.")
    if config.max_position_weight <= 0.0 or config.max_position_weight > 1.0:
        raise ValueError("max_position_weight must be in (0, 1].")
    if config.max_sector_weight <= 0.0 or config.max_sector_weight > 1.0:
        raise ValueError("max_sector_weight must be in (0, 1].")
    if config.lambda_risk < 0.0 or config.lambda_turnover < 0.0 or config.lambda_cost < 0.0:
        raise ValueError("lambda_risk, lambda_turnover, lambda_cost must be non-negative.")
    if config.min_target_weight < 0.0:
        raise ValueError("min_target_weight must be non-negative.")
    if config.miqp_time_limit_seconds <= 0.0:
        raise ValueError("miqp_time_limit_seconds must be strictly positive.")
    if config.miqp_relative_gap < 0.0:
        raise ValueError("miqp_relative_gap must be non-negative.")
    if config.miqp_candidate_pool_size <= 0:
        raise ValueError("miqp_candidate_pool_size must be strictly positive.")
