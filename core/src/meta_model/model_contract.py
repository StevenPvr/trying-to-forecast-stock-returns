from __future__ import annotations

DATE_COLUMN: str = "date"
TICKER_COLUMN: str = "ticker"
SPLIT_COLUMN: str = "dataset_split"
TRAIN_SPLIT_NAME: str = "train"
VAL_SPLIT_NAME: str = "val"
TEST_SPLIT_NAME: str = "test"

LEGACY_TARGET_COLUMN: str = "target_main"

INTRADAY_GROSS_RETURN_COLUMN: str = "target_intraday_open_to_close_log_return"
INTRADAY_NET_RETURN_COLUMN: str = "target_intraday_open_to_close_net_log_return"
INTRADAY_BENCHMARK_RETURN_COLUMN: str = "benchmark_intraday_open_to_close_net_log_return"
INTRADAY_EXCESS_RETURN_COLUMN: str = "target_intraday_open_to_close_excess_log_return"
INTRADAY_SECTOR_RESIDUAL_RETURN_COLUMN: str = (
    "target_intraday_open_to_close_sector_residual_log_return"
)
INTRADAY_CS_ZSCORE_TARGET_COLUMN: str = "target_intraday_open_to_close_net_cs_zscore"
INTRADAY_CS_RANK_TARGET_COLUMN: str = "target_intraday_open_to_close_net_cs_rank"

OVERNIGHT_NET_RETURN_COLUMN: str = "target_overnight_close_to_next_open_net_log_return"
SHORT_HOLD_NET_RETURN_COLUMN: str = "target_short_hold_1d_to_2d_net_log_return"
MEDIUM_HOLD_GROSS_RETURN_COLUMN: str = "target_medium_hold_3d_to_5d_log_return"
MEDIUM_HOLD_NET_RETURN_COLUMN: str = "target_medium_hold_3d_to_5d_net_log_return"

RAW_FORWARD_RETURN_COLUMN: str = MEDIUM_HOLD_GROSS_RETURN_COLUMN
BENCHMARK_FORWARD_RETURN_COLUMN: str = INTRADAY_BENCHMARK_RETURN_COLUMN
EXCESS_FORWARD_RETURN_COLUMN: str = INTRADAY_EXCESS_RETURN_COLUMN
SECTOR_RESIDUAL_FORWARD_RETURN_COLUMN: str = INTRADAY_SECTOR_RESIDUAL_RETURN_COLUMN
CS_ZSCORE_TARGET_COLUMN: str = INTRADAY_CS_ZSCORE_TARGET_COLUMN
CS_RANK_TARGET_COLUMN: str = INTRADAY_CS_RANK_TARGET_COLUMN
MODEL_TARGET_COLUMN: str = INTRADAY_CS_ZSCORE_TARGET_COLUMN
REALIZED_RETURN_COLUMN: str = INTRADAY_GROSS_RETURN_COLUMN
PRIMARY_PRODUCTION_LABEL_COLUMN: str = INTRADAY_NET_RETURN_COLUMN
SECONDARY_RESEARCH_LABEL_COLUMNS: tuple[str, ...] = (
    OVERNIGHT_NET_RETURN_COLUMN,
    SHORT_HOLD_NET_RETURN_COLUMN,
    MEDIUM_HOLD_NET_RETURN_COLUMN,
)

PREDICTION_COLUMN: str = "prediction"
SIGNAL_DATE_COLUMN: str = "signal_date"

EXECUTION_LAG_DAYS: int = 1
HOLD_PERIOD_DAYS: int = 0
LABEL_EMBARGO_DAYS: int = EXECUTION_LAG_DAYS + HOLD_PERIOD_DAYS

TARGET_PREFIXES: tuple[str, ...] = ("target_",)
BENCHMARK_PREFIXES: tuple[str, ...] = ("benchmark_",)
HIGH_LEVEL_CONTEXT_PREFIXES: tuple[str, ...] = ("hl_context_",)
EXACT_EXCLUDED_FEATURE_COLUMNS: frozenset[str] = frozenset({
    DATE_COLUMN,
    TICKER_COLUMN,
    SPLIT_COLUMN,
    LEGACY_TARGET_COLUMN,
    SIGNAL_DATE_COLUMN,
})


def is_excluded_feature_column(column_name: str) -> bool:
    if column_name in EXACT_EXCLUDED_FEATURE_COLUMNS:
        return True
    if column_name.startswith(HIGH_LEVEL_CONTEXT_PREFIXES):
        return True
    if column_name.startswith(TARGET_PREFIXES):
        return True
    return column_name.startswith(BENCHMARK_PREFIXES)
