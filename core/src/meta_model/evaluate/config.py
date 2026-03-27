from __future__ import annotations

from dataclasses import dataclass

TOP_FRACTION: float = 0.01
ACTION_CAP_FRACTION: float = 0.05
ALLOCATION_FRACTION: float = 0.05
GROSS_CAP_FRACTION: float = 1.0
HOLD_PERIOD_DAYS: int = 5
TRANSACTION_COST_RATE_PER_SIDE: float = 0.003
LONG_DAILY_FINANCING_RATE: float = 0.0002269
SHORT_DAILY_FINANCING_RATE: float = 0.0000231
TARGET_COLUMN: str = "target_main"
PREDICTION_COLUMN: str = "prediction"
DATE_COLUMN: str = "date"
TICKER_COLUMN: str = "ticker"
SPLIT_COLUMN: str = "dataset_split"
TRAIN_SPLIT_NAME: str = "train"
VAL_SPLIT_NAME: str = "val"
TEST_SPLIT_NAME: str = "test"
EXCLUDED_FEATURE_COLUMNS: frozenset[str] = frozenset({
    DATE_COLUMN,
    TICKER_COLUMN,
    TARGET_COLUMN,
    SPLIT_COLUMN,
})


@dataclass(frozen=True)
class BacktestConfig:
    top_fraction: float = TOP_FRACTION
    action_cap_fraction: float = ACTION_CAP_FRACTION
    allocation_fraction: float = ALLOCATION_FRACTION
    gross_cap_fraction: float = GROSS_CAP_FRACTION
    hold_period_days: int = HOLD_PERIOD_DAYS
    transaction_cost_rate_per_side: float = TRANSACTION_COST_RATE_PER_SIDE
    long_daily_financing_rate: float = LONG_DAILY_FINANCING_RATE
    short_daily_financing_rate: float = SHORT_DAILY_FINANCING_RATE
