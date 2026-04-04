from __future__ import annotations

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
ADV_PARTICIPATION_LIMIT: float = 0.05
TURNOVER_PENALTY_BPS: float = 5.0
MAX_SPREAD_BPS: float = 35.0
OPEN_HURDLE_BPS: float = 12.0
HOLD_HURDLE_BPS: float = 6.0
PBO_MAX_THRESHOLD: float = 0.20
DSR_MIN_THRESHOLD: float = 0.10
APPLY_PREDICTION_HURDLE: bool = False


@dataclass(frozen=True)
class BacktestConfig:
    top_fraction: float = TOP_FRACTION
    action_cap_fraction: float = ACTION_CAP_FRACTION
    allocation_fraction: float = ALLOCATION_FRACTION
    gross_cap_fraction: float = GROSS_CAP_FRACTION
    hold_period_days: int = HOLD_PERIOD_DAYS
    execution_lag_days: int = EXECUTION_LAG_DAYS
    benchmark_mode: str = BENCHMARK_MODE
    neutrality_mode: str = NEUTRALITY_MODE
    adv_participation_limit: float = ADV_PARTICIPATION_LIMIT
    turnover_penalty_bps: float = TURNOVER_PENALTY_BPS
    max_spread_bps: float = MAX_SPREAD_BPS
    open_hurdle_bps: float = OPEN_HURDLE_BPS
    hold_hurdle_bps: float = HOLD_HURDLE_BPS
    apply_prediction_hurdle: bool = APPLY_PREDICTION_HURDLE
    pbo_max_threshold: float = PBO_MAX_THRESHOLD
    dsr_min_threshold: float = DSR_MIN_THRESHOLD


def validate_backtest_config(config: BacktestConfig) -> None:
    if config.execution_lag_days < 0:
        raise ValueError("execution_lag_days must be non-negative.")
    if not config.benchmark_mode.strip():
        raise ValueError("benchmark_mode must be configured explicitly.")
    if not config.neutrality_mode.strip():
        raise ValueError("neutrality_mode must be configured explicitly.")
    if config.max_spread_bps <= 0.0:
        raise ValueError("max_spread_bps must be strictly positive.")
    if config.hold_hurdle_bps >= config.open_hurdle_bps:
        raise ValueError("hold_hurdle_bps must remain strictly below open_hurdle_bps.")
